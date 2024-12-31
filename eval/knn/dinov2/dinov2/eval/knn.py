# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
from functools import partial
import json
import logging
import os
import sys
from typing import List, Optional

import torch.distributed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print(sys.path)
import torch
from torch.nn.functional import one_hot, softmax

import dinov2.distributed as distributed
from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data.transforms import make_classification_eval_transform
from dinov2.eval.metrics import AccuracyAveraging, build_topk_accuracy_metric
from dinov2.eval.utils import ModelWithNormalize, evaluate, extract_features, instantiate_from_config, get_obj_from_str
import torch_xla.core.xla_model as xm
from torch_xla.amp import autocast
import torch_xla.distributed.xla_multiprocessing as xmp
import torchvision.datasets as datasets

logger = logging.getLogger("dinov2")
def get_base_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents or [],
        add_help=add_help,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Model configuration file",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        type=str,
        help="Output directory to write results and logs",
    )
    return parser

def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parents = parents or []
    setup_args_parser = get_base_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--train-dataset",
        dest="train_dataset_str",
        type=str,
        help="Training dataset",
    )
    parser.add_argument(
        "--val-dataset",
        dest="val_dataset_str",
        type=str,
        help="Validation dataset",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        help="Input size",
    )
    parser.add_argument(
        "--nb_knn",
        nargs="+",
        type=int,
        help="Number of NN to use. 20 is usually working the best.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature used in the voting coefficient",
    )
    parser.add_argument(
        "--gather-on-cpu",
        action="store_true",
        help="Whether to gather the train features on cpu, slower"
        "but useful to avoid OOM for large datasets (e.g. ImageNet22k).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size.",
    )
    parser.add_argument(
        "--n-per-class-list",
        nargs="+",
        type=int,
        help="Number to take per class",
    )
    parser.add_argument(
        "--n-tries",
        type=int,
        help="Number of tries",
    )
    parser.add_argument(
        "--autocast-dtype",
        type=str,
        default="torch.float",
        help="Autocast dtype",
    )
    parser.set_defaults(
        train_dataset_str="ImageNet:split=TRAIN",
        val_dataset_str="ImageNet:split=VAL",
        nb_knn=[10, 20, 100, 200],
        temperature=0.07,
        batch_size=256,
        n_per_class_list=[-1],
        n_tries=1,
        input_size=224,
    )
    return parser
def xm_gather_with_target_rank(tensor, output_list, dst):
    global_rank = xm.get_ordinal()
    xm.mark_step()
    gathered_tensor = xm.all_gather(tensor, pin_layout=False)
    if global_rank == dst:
        #gather tensor is of shape n*gloabl_size x dim, split it into n tensors of shape global_size x dim
        gathered_tensor = gathered_tensor.split(tensor.size(0), dim=0)
        output_list[:] = gathered_tensor
    return output_list
class KnnModule(torch.nn.Module):
    """
    Gets knn of test features from all processes on a chunk of the train features

    Each rank gets a chunk of the train features as well as a chunk of the test features.
    In `compute_neighbors`, for each rank one after the other, its chunk of test features
    is sent to all devices, partial knns are computed with each chunk of train features
    then collated back on the original device.
    """

    def __init__(self, train_features, train_labels, nb_knn, T, device, num_classes=1000):
        super().__init__()

        self.global_rank = distributed.get_global_rank()
        self.global_size = distributed.get_global_size()

        self.device = device
        self.train_features_rank_T = train_features.chunk(self.global_size)[self.global_rank].T.to(self.device)
        self.candidates = train_labels.chunk(self.global_size)[self.global_rank].view(1, -1).to(self.device)

        self.nb_knn = nb_knn
        self.max_k = max(self.nb_knn)
        self.T = T
        self.num_classes = num_classes

    def _get_knn_sims_and_labels(self, similarity, train_labels):
        topk_sims, indices = similarity.topk(self.max_k, largest=True, sorted=True)
        neighbors_labels = torch.gather(train_labels, 1, indices)
        return topk_sims, neighbors_labels

    def _similarity_for_rank(self, features_rank, source_rank):
        # Send the features from `source_rank` to all ranks
        broadcast_shape = torch.tensor(features_rank.shape).to(self.device)
        #torch.distributed.broadcast(broadcast_shape, source_rank)
        #broadcast_shape = xm.all_reduce("sum", broadcast_shape)
        if self.global_rank != source_rank:
            broadcasted = torch.zeros(*broadcast_shape, dtype=features_rank.dtype, device=self.device)
        else:
            broadcasted = features_rank
        #torch.distributed.broadcast(broadcasted, source_rank)
        broadcasted = xm.all_reduce("sum", broadcasted)
        # Compute the neighbors for `source_rank` among `train_features_rank_T`
        similarity_rank = torch.mm(broadcasted, self.train_features_rank_T)
        candidate_labels = self.candidates.expand(len(similarity_rank), -1)
        return self._get_knn_sims_and_labels(similarity_rank, candidate_labels)

    def _gather_all_knn_for_rank(self, topk_sims, neighbors_labels, target_rank):
        # Gather all neighbors for `target_rank`
        topk_sims_rank = retrieved_rank = None
        if self.global_rank == target_rank:
            topk_sims_rank = [torch.zeros_like(topk_sims) for _ in range(self.global_size)]
            retrieved_rank = [torch.zeros_like(neighbors_labels) for _ in range(self.global_size)]
        print(f"START GATHER: rank {self.global_rank} topk_sims: {topk_sims.shape},{topk_sims.device}, neighbors_labels: {neighbors_labels.shape},{neighbors_labels.device}")
        #torch.distributed.gather(topk_sims, topk_sims_rank, dst=target_rank)
        xm_gather_with_target_rank(topk_sims, topk_sims_rank, dst=target_rank)
        print(f"GATHER1: rank {self.global_rank} ")
        xm.rendezvous('knn-gather1')
        #torch.distributed.gather(neighbors_labels, retrieved_rank, dst=target_rank)
        print(f"GATHER2: rank {self.global_rank} ")
        xm_gather_with_target_rank(neighbors_labels, retrieved_rank, dst=target_rank)
        xm.rendezvous('knn-gather2')
        if self.global_rank == target_rank:
            # Perform a second top-k on the k * global_size retrieved neighbors
            topk_sims_rank = torch.cat(topk_sims_rank, dim=1)
            retrieved_rank = torch.cat(retrieved_rank, dim=1)
            results = self._get_knn_sims_and_labels(topk_sims_rank, retrieved_rank)
            print(f"TARGET RANK: rank {self.global_rank} results: {results[0].shape},{results[0].device}, {results[1].shape},{results[1].device}")
            return results
        return None

    def compute_neighbors(self, features_rank):
        for rank in range(self.global_size):
            xm.master_print(f"calculate knn for rank {rank}")
            topk_sims, neighbors_labels = self._similarity_for_rank(features_rank, rank)
            results = self._gather_all_knn_for_rank(topk_sims, neighbors_labels, rank)
            if results is not None:
                topk_sims_rank, neighbors_labels_rank = results
            xm.rendezvous('knn-compute')
        return topk_sims_rank, neighbors_labels_rank

    def forward(self, features_rank):
        """
        Compute the results on all values of `self.nb_knn` neighbors from the full `self.max_k`
        """
        assert all(k <= self.max_k for k in self.nb_knn)
        #xm.master_print(f"features_rank: {features_rank.shape}")
        topk_sims, neighbors_labels = self.compute_neighbors(features_rank)
        xm.mark_step()
        #xm.master_print(f"forward1: topk_sims: {topk_sims.shape}, neighbors_labels: {neighbors_labels.shape}")
        batch_size = neighbors_labels.shape[0]
        topk_sims_transform = softmax(topk_sims / self.T, 1)
        matmul = torch.mul(
            one_hot(neighbors_labels, num_classes=self.num_classes),
            topk_sims_transform.view(batch_size, -1, 1),
        )
        xm.mark_step()
        xm.master_print(f"matmul: {matmul.shape}")
        probas_for_k = {k: torch.sum(matmul[:, :k, :], 1) for k in self.nb_knn}
        return probas_for_k


class DictKeysModule(torch.nn.Module):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def forward(self, features_dict, targets):
        for k in self.keys:
            features_dict = features_dict[k]
        return {"preds": features_dict, "target": targets}


def create_module_dict(*, module, n_per_class_list, n_tries, nb_knn, train_features, train_labels):
    modules = {}
    mapping = create_class_indices_mapping(train_labels)
    for npc in n_per_class_list:
        if npc < 0:  # Only one try needed when using the full data
            full_module = module(
                train_features=train_features,
                train_labels=train_labels,
                nb_knn=nb_knn,
            )
            modules["full"] = ModuleDictWithForward({"1": full_module})
            continue
        all_tries = {}
        for t in range(n_tries):
            final_indices = filter_train(mapping, npc, seed=t)
            k_list = list(set(nb_knn + [npc]))
            k_list = sorted([el for el in k_list if el <= npc])
            all_tries[str(t)] = module(
                train_features=train_features[final_indices],
                train_labels=train_labels[final_indices],
                nb_knn=k_list,
            )
        modules[f"{npc} per class"] = ModuleDictWithForward(all_tries)

    return ModuleDictWithForward(modules)


def filter_train(mapping, n_per_class, seed):
    torch.manual_seed(seed)
    final_indices = []
    for k in mapping.keys():
        index = torch.randperm(len(mapping[k]))[:n_per_class]
        final_indices.append(mapping[k][index])
    return torch.cat(final_indices).squeeze()


def create_class_indices_mapping(labels):
    unique_labels, inverse = torch.unique(labels, return_inverse=True)
    mapping = {unique_labels[i]: (inverse == i).nonzero() for i in range(len(unique_labels))}
    return mapping


class ModuleDictWithForward(torch.nn.ModuleDict):
    def forward(self, *args, **kwargs):
        return_dict = {}
        for k, module in self.items():
            return_dict[k] = module(*args, **kwargs)
            xm.mark_step()
        return return_dict


def eval_knn(
    model,
    train_dataset,
    val_dataset,
    accuracy_averaging,
    nb_knn,
    temperature,
    batch_size,
    num_workers,
    gather_on_cpu,
    n_per_class_list=[-1],
    n_tries=1,
):
    model = ModelWithNormalize(model)

    logger.info("Extracting features for train set...")
    xm.master_print("Extracting features for train set...")
    train_features, train_labels = extract_features(
        model, train_dataset, batch_size, num_workers, gather_on_cpu=gather_on_cpu
    )   
    xm.master_print(f"Train features extracted, shape {train_features.shape}.")
    logger.info(f"Train features created, shape {train_features.shape}.")

    val_dataloader = make_data_loader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
        persistent_workers=True,
    )
    num_classes = train_labels.max() + 1
    metric_collection = build_topk_accuracy_metric(accuracy_averaging, num_classes=num_classes)

    #device = torch.cuda.current_device()
    device = xm.xla_device()
    partial_module = partial(KnnModule, T=temperature, device=device, num_classes=num_classes)
    knn_module_dict = create_module_dict(
        module=partial_module,
        n_per_class_list=n_per_class_list,
        n_tries=n_tries,
        nb_knn=nb_knn,
        train_features=train_features,
        train_labels=train_labels,
    )
    postprocessors, metrics = {}, {}
    for n_per_class, knn_module in knn_module_dict.items():
        for t, knn_try in knn_module.items():
            postprocessors = {
                **postprocessors,
                **{(n_per_class, t, k): DictKeysModule([n_per_class, t, k]) for k in knn_try.nb_knn},
            }
            metrics = {**metrics, **{(n_per_class, t, k): metric_collection.clone() for k in knn_try.nb_knn}}
    model_with_knn = torch.nn.Sequential(model, knn_module_dict)

    # ============ evaluation ... ============
    logger.info("Start the k-NN classification.")
    _, results_dict = evaluate(model_with_knn, val_dataloader, postprocessors, metrics, device)

    # Averaging the results over the n tries for each value of n_per_class
    for n_per_class, knn_module in knn_module_dict.items():
        first_try = list(knn_module.keys())[0]
        k_list = knn_module[first_try].nb_knn
        for k in k_list:
            keys = results_dict[(n_per_class, first_try, k)].keys()  # keys are e.g. `top-1` and `top-5`
            results_dict[(n_per_class, k)] = {
                key: torch.mean(torch.stack([results_dict[(n_per_class, t, k)][key] for t in knn_module.keys()]))
                for key in keys
            }
            for t in knn_module.keys():
                del results_dict[(n_per_class, t, k)]

    return results_dict


def eval_knn_with_model(
    model,
    output_dir,
    train_dataset_str="ImageNet:split=TRAIN",
    val_dataset_str="ImageNet:split=VAL",
    nb_knn=(10, 20, 100, 200),
    temperature=0.07,
    autocast_dtype=torch.float,
    accuracy_averaging=AccuracyAveraging.MEAN_ACCURACY,
    transform=None,
    gather_on_cpu=False,
    batch_size=256,
    num_workers=5,
    n_per_class_list=[-1],
    n_tries=1,
    input_size=224,
):
    transform = transform or make_classification_eval_transform(crop_size=input_size)

    train_dataset = datasets.ImageFolder(
        root=train_dataset_str,
        transform=transform,
    )
    val_dataset = datasets.ImageFolder(
        root=val_dataset_str,
        transform=transform,
    )
    xm.master_print(f"Train dataset: {train_dataset_str} with #samples: {len(train_dataset)}, Val dataset: {val_dataset_str} with #samples: {len(val_dataset)}")
    #with torch.cuda.amp.autocast(dtype=autocast_dtype):
    with autocast(device=xm.xla_device(), dtype=autocast_dtype):
        results_dict_knn = eval_knn(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            accuracy_averaging=accuracy_averaging,
            nb_knn=nb_knn,
            temperature=temperature,
            batch_size=batch_size,
            num_workers=num_workers,
            gather_on_cpu=gather_on_cpu,
            n_per_class_list=n_per_class_list,
            n_tries=n_tries,
        )

    results_dict = {}
    if distributed.is_main_process():
        for knn_ in results_dict_knn.keys():
            top1 = results_dict_knn[knn_]["top-1"].item() * 100.0
            top5 = results_dict_knn[knn_]["top-5"].item() * 100.0
            results_dict[f"{knn_} Top 1"] = top1
            results_dict[f"{knn_} Top 5"] = top5
            logger.info(f"{knn_} classifier result: Top1: {top1:.2f} Top5: {top5:.2f}")

    metrics_file_path = os.path.join(output_dir, "results_eval_knn.json")
    with open(metrics_file_path, "a") as f:
        for k, v in results_dict.items():
            f.write(json.dumps({k: v}) + "\n")

    if distributed.is_enabled():
        torch.distributed.barrier()
    return results_dict

from omegaconf import OmegaConf
def broadcast_test_tensor(tensor):
    #tensor = tensor.to(xm.xla_device())
    b_tensor = xm.all_reduce("sum", tensor)
    return b_tensor
def all_gather_test_tensor(tensor):
    tensor = tensor.to(xm.xla_device())
    g_tensor = xm.all_gather(tensor)
    return g_tensor
def main(rank, args):
    rank = int(os.environ.get('LOCAL_RANK', rank))
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '11451'
    torch.distributed.init_process_group("xla", init_method='xla://', rank=rank) # 4 is the number of TPU cores
    print(f"Rank {rank} initialized")
    #model, autocast_dtype = setup_and_build_model(args)
    config = args.config_file
    config = OmegaConf.load(config)
    model = instantiate_from_config(config)
    autocast_dtype = get_obj_from_str(args.autocast_dtype)
    eval_knn_with_model(
        model=model,
        output_dir=args.output_dir,
        train_dataset_str=args.train_dataset_str,
        val_dataset_str=args.val_dataset_str,
        nb_knn=args.nb_knn,
        temperature=args.temperature,
        autocast_dtype=autocast_dtype,
        accuracy_averaging=AccuracyAveraging.MEAN_ACCURACY,
        transform=None,
        gather_on_cpu=args.gather_on_cpu,
        batch_size=args.batch_size,
        num_workers=5,
        n_per_class_list=args.n_per_class_list,
        n_tries=args.n_tries,
        input_size=args.input_size,
    )
    return 0


if __name__ == "__main__":
    description = "DINOv2 k-NN evaluation"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    main(None, args)
    #xmp.spawn(main, args=(args,), start_method="fork")
    #sys.exit(main(args))

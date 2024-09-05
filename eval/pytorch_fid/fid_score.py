"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
use_TPU = os.path.exists('/dev/accel0') # if not then there is no TPU
if use_TPU:
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    # do a compilation cache
    cache_path = "/home/bytetriper/.cache/xla_compile/fid"
    xr.initialize_cache(cache_path, readonly=False)
try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x


#from inception import InceptionV3
from inception import InceptionV3
class InceptionWrapper(InceptionV3):

    def forward(self, inp):
        pred = super().forward(inp)[0]
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred = pred.reshape(pred.shape[0], -1)

        return pred

    def get_logits(self, inp):
        pred, logits = super().forward(inp, return_logits=True)
        pred = pred[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred = pred.reshape(pred.shape[0], -1)
        return pred, logits
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--batch-size","--bs", type=int, default=64, help="Batch size to use")
parser.add_argument(
    "--num-workers",
    type=int,
    help=(
        "Number of processes to use for data loading. " "Defaults to `min(8, num_cpus)`"
    ),
)
parser.add_argument(
    "--dims",
    type=int,
    default=2048,
    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
    help=(
        "Dimensionality of Inception features to use. "
        "By default, uses pool3 features"
    ),
)
parser.add_argument(
    "--use_cache",
    action="store_true",
    help="use cached act",
)
parser.add_argument(
    "--save_cache",
    action="store_true",
    help="save act as cache",
)
parser.add_argument(
    "path",
    type=str,
    nargs=2,
    help=("Paths to the generated images or " "to .npz statistic files"),
)


class npzDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = self.files[i]
        img = self.transforms(img)
        return img

def get_activations(
    file: np.ndarray,
    model: InceptionWrapper,
    batch_size=64, 
    dims=2048,
    device="cpu", 
    num_workers=1
):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : a numpy ndarray of images, shape (N, H, W, 3)
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > file.shape[0]:
        print(
            (
                "Warning: batch size is bigger than the data size. "
                "Setting batch size to data size"
            )
        )
        batch_size = file.shape[0]

    #dataset = ImagePathDataset(files, transforms=TF.ToTensor())
    # files is a npz file
    def data_transforms(img):
        return TF.ToTensor()(img)
        #return torch.tensor(img).permute(2, 0, 1).float() / 255.0 # hand-crafted transform for speed
    dataset = npzDataset(file, transforms=data_transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    pred_arr = np.empty((file.shape[0], dims))
    logits_arr = torch.empty((file.shape[0], 1008)) # InceptionV3 has 1008 classes
    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred, logits = model.get_logits(batch)
        logits = torch.nn.functional.softmax(logits, dim=-1)
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        #if pred.size(2) != 1 or pred.size(3) != 1:
        #    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        #pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred = pred.cpu().numpy()
        pred_arr[start_idx : start_idx + pred.shape[0]] = pred
        logits_arr[start_idx : start_idx + pred.shape[0]] = logits
        start_idx = start_idx + pred.shape[0]
        xm.mark_step() if use_TPU else None
    return pred_arr, logits_arr


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def calculate_Inception_Score(logits_arr:torch.Tensor, splits=10):
    scores = []
    ps = logits_arr
    num_samples = ps.shape[0]
    for j in range(splits):
        part = ps[(j * num_samples // splits):((j + 1) * num_samples // splits), :]
        kl = part * (torch.log(part) - torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
        kl = torch.mean(torch.sum(kl, 1))
        kl = torch.exp(kl)
        scores.append(kl.unsqueeze(0))
    scores = torch.cat(scores, 0)
    m_scores = torch.mean(scores).detach().cpu().numpy()
    m_std = torch.std(scores).detach().cpu().numpy()
    return m_scores, m_std
def calculate_activation_statistics(
    files, model, batch_size=64, dims=2048, device="cpu", num_workers=1
):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act, logits = get_activations(files, model, batch_size, dims, device, num_workers)
    return act, logits


def compute_statistics_of_path(path, model, batch_size, dims, device, num_workers=1, use_cache: bool = False, save_cache: bool = False):
    #path = pathlib.Path(path)
    #files = sorted(
    #    [file for ext in IMAGE_EXTENSIONS for file in path.glob("*.{}".format(ext))]
    #)
    assert path.endswith('.npz'), 'Only npz files are supported'
    potential_act_path = path.replace('.npz', '_act.npz') # see if we have already computed the activations
    if os.path.exists(potential_act_path) and use_cache:
        data =  np.load(potential_act_path)
        act, logits = data['act'], data['logits']
        logits = torch.tensor(logits)
        m = np.mean(act, axis=0)
        s = np.cov(act, rowvar=False)
    else:
        file = np.load(path)['arr_0'] # (N, H, W, 3), uint8
        act, logits = get_activations(
            file, model, batch_size, dims, device, num_workers
        )
        m = np.mean(act, axis=0)
        s = np.cov(act, rowvar=False)
        if save_cache:
            # save the activations as cache
            np.savez(potential_act_path,
                        act=act,
                        logits=logits.cpu().numpy()
            )
    return m, s, logits


def calculate_stat_given_paths(paths, batch_size, device, dims, num_workers=1, use_cache: bool = False, save_cache: bool = False):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError("Invalid path: %s" % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    #model = InceptionV3([block_idx]).to(device)
    model = InceptionWrapper([block_idx]).to(device)
    m1, s1, l1 = compute_statistics_of_path(
        paths[0], model, batch_size, dims, device, num_workers, use_cache, save_cache
    )
    m2, s2, l2 = compute_statistics_of_path(
        paths[1], model, batch_size, dims, device, num_workers, use_cache, save_cache
    )
    #fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    fid_value = frechet_distance(m1, s1, m2, s2)
    IS_value, IS_std = calculate_Inception_Score(l2) # l2 is the test batch
    return fid_value, IS_value, IS_std



def main():
    args = parser.parse_args()

    device = xm.xla_device() if use_TPU else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    fid_value, IS_value, IS_std = calculate_stat_given_paths(
        args.path, args.batch_size, device, args.dims, num_workers, args.use_cache, args.save_cache
    )
    print("FID: ", fid_value)
    print("IS: ", IS_value, "+-", IS_std)
    


if __name__ == "__main__":
    main()

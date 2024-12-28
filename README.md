# MAE-DiT: also a general ddp framework for training and testing vision models

## Quick Start

### Basic Functionality

This codebase support:

- Training and testing of Stage1 and Stage2 models
- Customized dataset, model, loss
- Simple config-based customization
- Scalable TPU training
- Built-in gan loss, perceptual loss, online FID evaluation
- Wandb logging w/ visualization

### Installation

We strongly suggest to use `torch_xla >= 2.5.0` to avoid possible function bug when ` XLA_DISABLE_FUNCTIONALIZATION=1` is used for acceleration.

```bash
pip install torch~=2.5.0 torch_xla[tpu]~=2.5.0 -f https://storage.googleapis.com/libtpu-releases/index.html
```

and

```bash
pip install -r requirements.txt
```

If you want to use the PSNR/SSIM metric under `eval/`, you need to additionally do:

`sudo apt-get install ffmpeg libsm6 libxext6 # required for opencv (cv2)`

and `pip install opencv-python`

### Basic Usage

A main illustration of the structure of the repo is:

```bash
.
├── configs
│   ├── xxx/xxx.yaml # config file for training/testing

|── ckpt_gcs # (optional) for storing checkpoints on GCS using GCSFuse

|── data # often soft-linked to PD
    |── imagenet # (optional) for storing imagenet data
    |── xxx # (optional) for storing other datasets
|── rqvae # main codebase
    |── img_datasets # dataset classes
    |── models # implementation of stage1/2 models
        |── interfaces.py # abstract classes for models
    

|── eval # evaluation scripts
    |── pytorch_fid/ # FID evaluation code for torch_XLA
    |── psnr_ssims.py # PSNR/SSIM evaluation code

|── gpu_eval # evaluation scripts for GPU
    |── fid/ # standard FID evaluation code (ADM suite)
```

### Abstractions

The codebase is designed to be modular and extensible. The main pipeline is:

**Dataset**

All customized dataset instances should inherit from `torch.utils.data.Dataset` and re-implement the `__getitem__` to return a `LabeledImageData` instance (defined in `rqvae/img_datasets/interfaces.py`). Optionally, you can also implement the `collate_fn` to customize the batch collation behavior. If not defined, it will used a default collation function `rqvae.img_datasets.LabeledImageDatasetWrapper.default_collate_fn`.

To properly use the dataset, you need to modify `rqvae/img_datasets/__init__.py` to include your dataset class (sorry for a stack of if/else statements, I will fix it to config-based soon).

Likewise, you also need to modify `rqvae/img_datasets/transforms.py` to include your customized transforms. (again sorry for the stack of if/else statements)

`LabeledImageData` is a `@dataclass` with the following fields:

- `img`: `torch.Tensor` (optional)
- `condition`: `Any` (optional)
- `img_path`: `str` (optional)
- `additional_attr`: `Optional[str]` (optional)


**Stage1 Model**

The main idea for Stage1 Model is that it encodes the input into a latent representation and decodes it back to a reconstruction.
Stage1 Model's behavior is defined (in `rqvae/models/interfaces.py`) as:

- `forward`: it accepts a `LabeledImageData` as input and returns a `Stage1ModelOutput` as output in forward pass (encoding and decoding)
- `encode` it accepts a `LabeledImageData` and returns a `Stage1Encodings` 
- `decode` it accepts a `Stage1Encodings`  and returns a `Stage1ModelOutput`
- `compute_loss` it accepts a `Stage1ModelOutput`(reconstruction) and a `LabeledImageData`(input) and returns a dict for loss.


`Stage1ModelOutput` is a `@dataclass` with the following fields:
- `xs_recon`: `torch.Tensor` (reconstruction)
- `additional_attr`: `dict` (optional) for storing possible additional attributes


`Stage1Encodings` is a `@dataclass` with the following fields:
- `zs`: `torch.Tensor` (latent representation)
- `additional_attr`: `dict` (optional) for storing possible additional attributes like `mu`, `logvar` in VAE


For loss, it's a dict (sorry for no abstraction here, it should be one), with three keys:
```python
{
    'loss_total': torch.Tensor, # total loss
    'loss_recon': torch.Tensor, # reconstruction loss like L1
    'loss_latent': torch.Tensor, # latent loss like KL divergence
}
```
you should always set `loss_total` to be the sum of `loss_recon` and `loss_latent` in `compute_loss`. The actual total loss in training will be `loss_total`.

**Connector**

The connector is a simple class that connects the Stage1 and Stage2 models. For example, you may want to do a reshape to reshape a 1D stage1 latent to 2D for stage2, and reshape it back for decoding. The connector is defined (in `rqvae.models.interfaces.base_connector`) as:

- `forward`: it accepts a `Union[Stage1Encodings, Stage2ModelOutput]` and returns a `Stage1Encodings`. This function is called after Stage1 Model's `encode` and before Stage2 Model's `forward`.
- `reverse`: it accepts a `Union[Stage1Encodings, Stage2ModelOutput]` and returns a `Stage1Encodings`. This function is called after Stage2 Model's `forward` and before Stage1 Model's `decode`.

By default we have a `base_connector` that does nothing but simply return the input. Note that this will return a `Stage2ModelOutput` for `forward` and `reverse` if the input is a `Stage2ModelOutput`, which is a bit invalid. So if you're using a `base_connector`, you should make sure your Stage1 Model can accept a `Stage2ModelOutput` as input.


**Stage2 Model**

The main idea for Stage2 Model is that it accepts the latent from stage1 and the input data, learn something from them, then output a latent for decoding in generation. 
Stage2 Model's behavior is defined (in `rqvae/models/interfaces.py`) as:

- `forward`: it accepts a `Stage1Encodings` and a `LabeledImageData` as input and returns a `Stage2ModelOutput` as output in forward pass
- `compute_loss`: it accepts a `Stage1Encodings`, a `Stage2ModelOutput` and a `LabeledImageData` and returns a dict for loss. Only `loss_total` is required here.
- `infer` (generation): it accepts a `LabeledImageData` (usually not needed as we're doing noise-to-image generation) and returns a `Stage2ModelOutput` for decoding.


`Stage2ModelOutput` is a `@dataclass` with the following fields:
- `zs_pred`: `torch.Tensor` (generated latent)
- `zs_degraded`: `torch.Tensor` (degraded latent, for example, noised), often not needed
- `additional_attr`: `dict` (optional) for storing possible additional attributes


**Stage1 Training**

Basic idea for stage1 training is an AE like training recipe. It feeds data into the Stage1 Model, get a loss from the loss function of the model, and additionally do a LPIPS and GAN loss. For one iteration, we update the optimizer for model once and the optimizer for the discriminator once. The training is defined in `rqvae/trainers/stage1_trainer.py`.

Pipeline:

- Call dataloader to get a batch of data: `LabeledImageData`
- Call Stage1 Model's `forward` to get `Stage1ModelOutput`
- Call Stage1 Model's `compute_loss` with `Stage1ModelOutput` and `LabeledImageData` to get loss (`loss_total`, `loss_recon`, `loss_latent`)
- Extract `xs_recon` from `Stage1ModelOutput` and `img` from `LabeledImageData` for LPIPS loss and GAN loss
- use `loss_total + loss_lpips * lpips_weight + loss_gan * gan_weight` as the total loss for Stage1 Model
- Update the optimizer and scheduler for Stage1 Model
- Do another forward pass to get `Stage1ModelOutput` (for GAN)
- Calculate the GAN loss for the discriminator 
- Update the optimizer for the discriminator

**note** that we assume the `xs_recon`, `img` are in the range of [0, 1] to properly calculate GAN and LPIPS, you should do the normalization and de-normalization in the stage1 model.

**Stage2 Training**

We basically follow a diffusion training recipe for Stage2 Model. In every epoch we do a encoding to get the latent, then feed the Stage2 Model with the latent and the input data to get a prediction. We then calculate the loss and update the optimizer. The training is defined in `rqvae/trainers/stage2_trainer.py`.

Pipeline:

- Call dataloader to get a batch of data: `LabeledImageData`
- Call Stage1 Model's `encode` to get `Stage1Encodings`
- Call Connector's `forward` to get `Stage1Encodings` for Stage2 Model
- Call Stage2 Model's `forward` with `Stage1Encodings` and `LabeledImageData` to get `Stage2ModelOutput`
- Call Stage2 Model's `compute_loss` with `Stage1Encodings`, `Stage2ModelOutput` and `LabeledImageData` to get loss (`loss_total`)
- Update the optimizer and scheduler for Stage2 Model

(- Call `inference` every 0.5 epoch for visualization in wandb)

**Online FID Eval**

We break down the FID eval into:
- InceptionV3 feature extraction, which is done in parallel on pod
- FID calculation, which is done on the master node after a feature all_reduce

To use the online FID eval, you need to first provide a npz file (by `--fid_gt_act_path` in the training script) containing the InceptionV3 features for the **Ground Truth** data (how to ? TBD). Then set `--do_online_eval` to `True` in the training script. The FID will be logged to wandb for every eval step and the feature extracted will be saved.

**Stage1 Inference**

We use the test set of the dataset as the input for Stage1 Inference, and call `forward` to get the reconstruction. We'll call `get_recon_imgs` for image and reconstructions to get the final visualization (usually you just set it to return the input or simply do a clipping). 

**Stage2 Inference**
**TBD**

**Configurations**

The config file is defined as a `yaml` file in `configs/` (or any other directory you want). The config file should contain the following fields:

**TBD**

Currently, please see `configs/imagenet256/stage1/klvae/f_16.yaml` for an example for Stage1 and `configs/imagenet256/stage2/pixel/SID.yaml` for an example for Stage2.

For training we'll need all the fields, but for inference we only need the definition of dataset and model. Additional fields are simply ignored.
### Training & Testing

To train your Stage1 Model:

```bash
./train_stage1.sh [wand_project_name] [save_directory] [config_path] [world_size] [wandb_run_name] [resume_ckpt_path(optional)] [resume_wandb_id(optional)]
```

This will save the checkpoints to `save_directory/wand_project_name/wandb_run_name_{TIME}/` and log to wandb with the name `wandb_run_name_{TIME}` in the project `wand_project_name`. By default, `--do_online_eval` is set to `True` for Stage1 training, so run `./train_stage1woeval.sh` with same params if you don't want to do online FID eval.

To do an inference for Stage1 Model (the input will be the test set):

```bash
./recon_stage1.sh [save_directory] [ckpt_path] [config] [world_size] [wandb_run_name] [resume_ckpt_path(optional)] [resume_wandb_id(optional)]
```
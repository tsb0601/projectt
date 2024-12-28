# MAE-DiT: also a general ddp framework for training and testing vision models

### Quick Start

#### Basic Functionality

This codebase support:

- Training and testing of Stage1 and Stage2 models
- Customized dataset, model, loss
- Simple config-based customization
- Scalable TPU training
- Built-in gan loss, perceptual loss, online FID evaluation
- Wandb logging w/ visualization

#### Installation

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

#### Basic Usage

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

#### Abstractions

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

The connector is a simple class that connects the Stage1 and Stage2 models. For example, you may want to do a reshape to reshape a 1D stage1 latent to 2D for stage2, and reshape it back for decoding. The connector is defined in `rqvae.models.interfaces.py` as:

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



#### Customize your model

To customize your model `MyModel`, you need to:

- Implement `MyModel` in `rqvae/models/`
- Follow the specific structure of `rqvae/models/interfaces.py` to implement the model
- Add the corresponding config file in `configs/` to specify the model
- Done

More specifically, to implement a Stage1 Model:


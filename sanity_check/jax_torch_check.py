"""
this file requires both jax and torch to be installed
"""
import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # add parent directory to path
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../JAX'))) # JAX directory
from PIL import Image
import torch
from rqvae.models.utils import instantiate_from_config
from rqvae.models import create_model
from rqvae.img_datasets.interfaces import LabeledImageData
from omegaconf import OmegaConf
import numpy as np
import jax.numpy as jnp
import jax
#jax.config.update('jax_default_matmul_precision', 'float64')  # default, high, fastest
from torchvision.transforms import ToTensor, ToPILImage
from convert_weight import get_model_from_jax_weight, jax_model_encode, jax_model_decode
config_path = sys.argv[1]
jax_path = sys.argv[2]
print('config_path:',config_path)
print('save_path:',jax_path)
def create_torch_model(config_path):
    with torch.no_grad():
        config = OmegaConf.load(config_path).arch
        stage1_model_wrapper, _  = create_model(config, is_master=True) # load ckpt if available
        stage1_model_wrapper
        print('stage1 model:',stage1_model_wrapper)
        stage1_model = stage1_model_wrapper.stage_1_model
    return stage1_model
def create_jax_model(jax_weight_path):
    model, params = get_model_from_jax_weight(jax_weight_path)  
    return model, params
def loss_fn(x, y):
    return (np.abs(x-y)).mean()
def single_test(imgs:np.ndarray, torch_model, jax_model, jax_params)-> float:
    """
    imgs should be a numpy array of shape (n_samples, 3, 256, 256), in [0, 1]
    """
    torch_imgs = torch.tensor(imgs).permute(0, 3, 1, 2).float()
    torch_input = LabeledImageData(img=torch_imgs)
    jax_input = jnp.array(imgs)
    
    # torch forward
    with torch.no_grad():
        torch_encodings = torch_model.encode(torch_input)
        torch_latent = torch_encodings.zs
        torch_output = torch_model.decode(torch_encodings).xs_recon # (n_samples, 3, 256, 256)
    with jax.default_matmul_precision('highest'):
    # jax forward
        jax_latent, _, _ = jax_model.apply(jax_params, jax_input, train= False, rngs = {"dropout":jax.random.PRNGKey(0)}, method = jax_model_encode)
        _, jax_output = jax_model.apply(jax_params, jax_latent,None, train= False, rngs = {"dropout":jax.random.PRNGKey(0)}, method = jax_model_decode)
    # calculate loss
    np_torch_output = torch_output.permute(0, 2, 3, 1).numpy() # (n_samples, 256, 256, 3)
    np_jax_output = np.array(jax_output) # (n_samples, 3, 256, 256)
    loss = loss_fn(np_torch_output, np_jax_output)
    jax_latent = np.array(jax_latent)
    torch_latent = torch_latent.numpy()
    latent_loss = loss_fn(jax_latent, torch_latent)
    return loss, latent_loss
from tqdm import tqdm
def main(config_path, jax_weight_path, test_iter: int = 100):
    torch_model = create_torch_model(config_path)
    jax_model, jax_params = create_jax_model(jax_weight_path)
    print('jax_model:',jax_model)
    #print('jax_params:',jax_params)
    avg_loss = 0
    avg_latent_loss = 0
    for i in tqdm(range(test_iter)):
        imgs = np.random.rand(100, 256, 256, 3)
        # repeat to batch size 2
        loss , latent_loss= single_test(imgs, torch_model, jax_model, jax_params)
        avg_loss += loss
        avg_latent_loss += latent_loss
    print('loss:',avg_loss / test_iter, 'latent_loss:', avg_latent_loss / test_iter)
    return

if __name__ == '__main__':
    main(config_path, jax_path)
    
    
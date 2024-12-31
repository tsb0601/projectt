import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image
import torch
from rqvae.models.utils import instantiate_from_config
from torchvision.transforms import ToTensor, ToPILImage
from rqvae.img_datasets.interfaces import LabeledImageData
from rqvae.models import create_model
from torch_xla.amp import autocast
import torch_xla.core.xla_model as xm
from rqvae.models.connectors import base_connector
from rqvae.models.interfaces import *
import sys
import os
from check_utils import *
from omegaconf import OmegaConf
def count_params(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)
config_path = sys.argv[1]
# accept a tuple of image size from sys.argv[2], if not provided, default to (256, 256, 3)
im_size = tuple(map(int, sys.argv[2].split(','))) if len(sys.argv) > 2 else (256, 256)
assert os.path.isfile(config_path), f'Invalid config path {config_path}'
with torch.no_grad():
    config = OmegaConf.load(config_path).arch
    stage2_model_wrapper, _  = create_model(config, is_master=True) # load the model on the master
    stage2_model_wrapper:Stage2ModelWrapper
    print('stage2_model_wrapper:', stage2_model_wrapper)
    param, trainable_param = count_params(stage2_model_wrapper)
    print(f"Total params: {param/1e6:.2f}M, Trainable params: {trainable_param/1e6:.2f}M")
    
def test_all(stage2_model_wrapper:Stage2ModelWrapper, im_size:tuple):
    stage1_model = stage2_model_wrapper.stage_1_model
    connector = stage2_model_wrapper.connector
    stage2_model = stage2_model_wrapper.stage_2_model
    image = get_default_image(im_size)
    print(image.shape, image.min(), image.max())
    #image = (image * 2) - 1.
    #noise = torch.arange(patch_num).unsqueeze(0).expand(image.shape[0], -1)
    data = LabeledImageData(img=image)
    print("=" * 10, 'testing wrapper forward', "=" * 10)
    forward_output = stage2_model_wrapper.forward(data)[1]
    print(forward_output.zs_pred.shape, forward_output.zs_pred.min(), forward_output.zs_pred.max())
    print("=" * 10, 'testing stage1 get last layer', "=" * 10)
    last_layer = stage1_model.get_last_layer()
    print(last_layer.shape, last_layer.dtype)
    print("=" * 10, 'testing stage2 get last layer', "=" * 10)
    last_layer = stage2_model_wrapper.get_last_layer()
    print(last_layer.shape, last_layer.dtype)
    print("=" * 10, 'testing encoding', "=" * 10)
    latent_output = stage1_model.encode(data)
    print('encoded zs:', latent_output.zs.shape, latent_output.zs.mean(), latent_output.zs.std())
    print("=" * 10, 'testing connector', "=" * 10)
    connected_latent_output = connector.forward(latent_output)
    print('connected zs:', connected_latent_output.zs.shape, connected_latent_output.zs.mean(), connected_latent_output.zs.std())
    if connector.bn is not None:
        # we need to normalize the latent space
        print("=" * 10, 'testing connector normalization', "=" * 10)
        normalized_output = connector.normalize(connected_latent_output)
        print('normalized zs:', normalized_output.zs.shape, normalized_output.zs.mean(), normalized_output.zs.std())
        print("=" * 10, 'testing connector unnormalization', "=" * 10)
        unnormalized_output = connector.unnormalize(normalized_output)
        print('unnormalized zs:', unnormalized_output.zs.shape, unnormalized_output.zs.mean(), unnormalized_output.zs.std())
        if stage2_model_wrapper.do_normalize is False:
            # send a warning
            print("=" * 10, 'WARNING: connector has normalization but stage2 model has do_normalize=False', "=" * 10)
    else:
        print("=" * 10, 'connector has no batchnorm, skipping normalization', "=" * 10)
    print("=" * 10, 'testing forward', "=" * 10)
    forward_output = stage2_model.forward(connected_latent_output, data)
    print('forward zs_pred:', forward_output.zs_pred.shape, forward_output.zs_pred.mean(), forward_output.zs_pred.std())
    print("=" * 10, 'testing reverse', "=" * 10)
    reverse_output = connector.reverse(forward_output)
    print('reverse zs:', reverse_output.zs.shape, reverse_output.zs.mean(), reverse_output.zs.std())
    print("|reverse zs - latent zs|:", torch.abs(reverse_output.zs - latent_output.zs).mean())
    print("=" * 10, 'testing decoding (on stage2 output)', "=" * 10)
    recon_output = stage1_model.decode(reverse_output)
    recon = recon_output.xs_recon
    print(recon.shape, recon.min(), recon.max())
    print("=" * 10, 'testing decoding (on stage1 encoding)', "=" * 10)
    reverse_output = connector.reverse(connected_latent_output)
    recon_output = stage1_model.decode(reverse_output)
    recon = recon_output.xs_recon
    print(recon.shape, recon.min(), recon.max())
    print("=" * 10, 'testing stage1 loss', "=" * 10)
    try:
        loss = stage1_model.compute_loss(recon_output, data)['loss_total']
        print(loss)
    except NotImplementedError:
        print('loss not implemented')
    print("=" * 10, 'testing stage2 loss', "=" * 10)
    loss = stage2_model.compute_loss(connected_latent_output ,forward_output, data)['loss_total']
    print(loss)
    do_infer = sys.argv[3] if len(sys.argv) > 3 else False
    print("=" * 10, f'testing stage2 infer {"(skipped)" if not do_infer else ""}', "=" * 10)
    if do_infer:
        with autocast(device=xm.xla_device()):
            generated_output = stage2_model.infer(data)
        print(generated_output.zs_pred.shape, generated_output.zs_pred.min(), generated_output.zs_pred.max())
    print("=" * 10, 'all set!', "=" * 10)
print("=" * 10, 'testing stage2 model in eval mode', "=" * 10)
stage2_model_wrapper.eval()
test_all(stage2_model_wrapper, im_size)
print("=" * 10, 'testing stage2 model in train mode', "=" * 10)
stage2_model_wrapper.train()
test_all(stage2_model_wrapper, im_size)
print("=" * 10, 'all set!', "=" * 10)
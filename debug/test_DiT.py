import os
import sys
#set working directory to be the parent directory of the current file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rqvae.models.DiT import DiT_Stage2
import torch
from rqvae.img_datasets.interfaces import LabeledImageData
import torch_xla.core.xla_model as xm
from rqvae.models.interfaces import Stage1Encodings, Stage2ModelOutput
def test_DiT():
    model = DiT_Stage2(input_size=16, num_classes=1000, hidden_size=256, depth=4, in_channels=768)
    print(model)
    device = xm.xla_device()
    model = model.to(device)
    inputs = torch.randn(1, 768, 16, 16).to(device)
    labels = torch.ones((1,)).long().to(device) * 1000 # drop label
    stage1_encodings = Stage1Encodings(zs=inputs)
    data = LabeledImageData(img=inputs, condition=labels)
    data._to(device)
    test_forward_output = model(stage1_encodings, data)
    test_loss = model.compute_loss(stage1_encodings, test_forward_output, data)
    loss_total = test_loss['loss_total']
    print(f'loss_total: {loss_total}')
    test_infer_output = model.infer(data)
    sample = test_infer_output.zs_pred
    print(f'sample shape: {sample.shape}')

if __name__ == '__main__':
    test_DiT()
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from rqvae.models.mae_pt import Stage1MAE_For_Probing_PT

model = Stage1MAE_For_Probing_PT('vit_base_patch16',1000, False,  './ckpt_gcs/model_zoo/mae_base_224_pt/mae_pretrain_vit_base.pth')
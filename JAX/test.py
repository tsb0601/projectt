from model import VisionTransformer
from configs import get_base_config

config = get_base_config()
model = VisionTransformer(**config, num_classes=1000)

print(model)

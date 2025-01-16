from huggingface_hub import HfApi
from huggingface_hub import login

# Login to Hugging Face (you'll need to get a token from https://huggingface.co/settings/tokens)
login()

# Initialize the API
api = HfApi()

# Upload the file to your dataset
api.upload_file(
    path_or_fileobj="/home/tsb/projectt/inception_features/val_256_act_norm.npz",
    path_in_repo="val_256_act_norm.npz",
    repo_id="tsbpp/4tpu",
    repo_type="dataset"
)
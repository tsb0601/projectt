import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
# Simulated base directory where images are stored
base_directory = './'  # Replace with the actual base directory if needed
from PIL import Image
# Define transformer sizes and the number of groups
transformer_sizes = ['data/imagenet/val_256','ckpt/dino_decoder_l1_lpips_r', 'ckpt/clip_decoder_l1_lpips_r', 'ckpt/klvae_ema_recon', 'ckpt/mae_256_r/lpips_l1']
num_rows = 4  # Number of groups
num_columns = len(transformer_sizes)  # Number of transformer sizes

# Generate file paths dynamically
simulated_images = []
for row in range(num_rows):
    group = []
    for i, size in enumerate(transformer_sizes):
        # Generate file path based on size and group
        file_path = os.path.join(base_directory, size, f"ILSVRC2012_val_0000000{row+3}.png")
        group.append(file_path)
    simulated_images.append(group)

# Create the plot with updated figure size
fig, axs = plt.subplots(1,1)
group_imgs = []
# Loop through each group and each transformer size
for row in range(num_rows):
    imgs = []
    for col in range(num_columns):
        img_path = simulated_images[row][col]  # Access the image path for this group and column
        print(f"Loading image: {img_path}")
        if os.path.exists(img_path):  # Check if the image file exists
            img = mpimg.imread(img_path)
            # to rgb
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            # resize to 256x256
            # to PIL
            img = Image.fromarray((img * 255).astype(np.uint8))
            img = img.resize((256, 256))
            # to np
            img = np.array(img)
            print(f"Image shape: {img.shape}")
            imgs.append(img)
        else:
            raise FileNotFoundError(f"Image not found: {img_path}")
    # concatenate the images horizontally
    # to np
    imgs = np.array(imgs)
    # concatenate the images horizontally
    group_img = np.concatenate(imgs, axis=0)
    group_imgs.append(group_img)
# concatenate the images vertically
group_imgs = np.array(group_imgs)
# concatenate the images vertically
final_img = np.concatenate(group_imgs, axis=1).astype(np.uint8)
# transpose shape
# Display the final image
plt.imshow(final_img)
plt.axis('off')  # Turn off axis
# Set a title with an arrow and larger font size
#fig.suptitle("Increasing Transformer Size →", fontsize=12, fontweight='bold', y=0.95)
plt.tight_layout()
# Adjust layout to remove gaps within groups and reduce gaps between groups
plt.savefig('visuals/transformer_sizes.pdf', dpi= 1000,  bbox_inches="tight")
print("Plot saved as 'visuals/transformer_sizes.png'")

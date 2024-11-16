from json import load
import matplotlib.pyplot as plt
import numpy as np
from math import log10
from dit_flops import get_dit_flops
from load_metrics_data import load_metric_data
from utils import colors
dit_gflops = {
    (model_config, patch_size): get_dit_flops(model_config, patch_size)
    for model_config in ["XL", "L", "B"]
    for patch_size in [2]
}
min_dit_gflops = log10(min(dit_gflops.values()))
max_dit_gflops = log10(max(dit_gflops.values()))
save_dir = './visuals/fid'
import os
os.makedirs(save_dir, exist_ok=True)
def marker_size(config, patch_size, min_val=6, max_val=8):
    config = "S" if config == "s" else config
    config = "XL" if config == "H" else config
    key = (config, patch_size)
    if key not in dit_gflops:
        raise NotImplementedError(f"GFLOPs not implemented for {key}")
    return min_val + (log10(dit_gflops[key]) - min_dit_gflops) / (max_dit_gflops - min_dit_gflops) * (max_val - min_val)
import matplotlib.pyplot as plt
import numpy as np
def fid_plot_func(dit_fids, mae_fids):
    # Example data
    
    # Plot
    fig, ax = plt.subplots()

    line_styles = ['-', '-', '-']
    labels = ['DiT-VAE-XL', 'DiT-VAE-L', 'DiT-VAE-B']
    maelabels = ['DiT-MAE-XL', 'DiT-MAE-L', 'DiT-MAE-B']
    xs = [1,2,3]
    markers = 'o'
    line_widths = 4
    marker_size = 8
    colors1 = [colors[1], colors[4]]
    print('dit_fids:', dit_fids)
    print('mae_fids:', mae_fids)
    # Plot each line
    plt.plot(xs,dit_fids, linestyle='-', color=colors1[0], marker=markers, linewidth=line_widths, markersize=marker_size, label='V+DiT')
    plt.plot(xs,mae_fids, linestyle='-', color=colors1[1], marker=markers, linewidth=line_widths, markersize=marker_size, label='M+DiT')
    # Customize the plot
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("FID-50k", fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    # ranging from 100k to 400k
    #ax.set_xlim(100000, 400000)
    # Set x-ticks in scientific notation or custom formatting
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['B', 'L', 'XL'])
    ax.set_xlim(0.5, 3.5)
    #ax.set_title(r"$\it{DiT\ with\ MAE\ works\ better!}$", fontsize=16, pad = 15)
    # Display or save the plot
    #plt.show()
    fig.savefig("./visuals/fid/fid_scale.pdf", dpi=1000, bbox_inches="tight")


def main():
    # Load the data
    fids = load_metric_data('fid50k', True)
    mae_fid_data = {
        key: fids[key][0][1] for key in fids.keys() if key in ['MAE-XL', 'MAE-L', 'MAE-B']
    }
    mae_fid_data = [mae_fid_data['MAE-B'], mae_fid_data['MAE-L'], mae_fid_data['MAE-XL']]
    dit_fid_data = {
        key: fids[key][0][1] for key in fids.keys() if key in 
        ['DiT-XL', 'DiT-L', 'DiT-B']
    }
    dit_fid_data = [dit_fid_data['DiT-B'], dit_fid_data['DiT-L'], dit_fid_data['DiT-XL']]
    # Plot the data
    fid_plot_func(dit_fid_data, mae_fid_data)
    print('saved plot to ./visuals/fid/fid_scale.pdf')
if __name__ == "__main__":
    main()
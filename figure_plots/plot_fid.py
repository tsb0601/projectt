from json import load
import matplotlib.pyplot as plt
import numpy as np
from math import log10
from dit_flops import get_dit_flops
from load_metrics_data import load_metric_data
from utils import colors
dit_gflops = {
    (model_config, patch_size): get_dit_flops(model_config, patch_size)
    for model_config in ["XL", "L"]
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
    labels = ['DiT-VAE-XL', 'DiT-VAE-L']
    maelabels = ['DiT-MAE-XL', 'DiT-MAE-L']
    markers = ['o', 'o', 'o']
    line_widths = [2, 3, 4]
    marker_sizes = [marker_size('XL', 2), marker_size('L', 2)]
    colors2 = [colors[1],colors[0]]
    colors1 = [colors[4], colors[3]]
    print('dit_fids:', dit_fids)
    print('mae_fids:', mae_fids)
    # Plot each line
    for i, (dit_fid_name, color, lw, ms, label) in enumerate(zip(dit_fids, colors1, line_widths, marker_sizes, labels)):
        dit_fid = dit_fids[dit_fid_name]
        print('dit_fid:', dit_fid)
        x_fid = [x[0] for x in dit_fid]
        y_fid = [x[1] for x in dit_fid]
        ax.plot(x_fid, y_fid, linestyle=line_styles[i], color=color, marker=markers[i], linewidth=lw, markersize=ms, label=label)
    for i, (mae_fid_name, color, lw, ms, label) in enumerate(zip(mae_fids, colors2, line_widths, marker_sizes, maelabels)):
        mae_fid = mae_fids[mae_fid_name]
        x_fid = [x[0] for x in mae_fid]
        y_fid = [x[1] for x in mae_fid]
        ax.plot(x_fid, y_fid, linestyle=line_styles[i], color=color, marker=markers[i], linewidth=lw, markersize=ms, label=label)
    # Customize the plot
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("FID-50k", fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    # ranging from 100k to 400k
    ax.set_xlim(100000, 400000)
    # Set x-ticks in scientific notation or custom formatting
    ax.set_xticks([100000, 200000, 300000, 400000])
    ax.set_xticklabels(['100k', '200k', '300k', '400k'])
    #ax.set_title(r"$\it{DiT\ with\ MAE\ works\ better!}$", fontsize=16, pad = 15)
    # Display or save the plot
    #plt.show()
    fig.savefig("./visuals/fid/training_steps_plot.pdf", dpi=1000, bbox_inches="tight")


def main():
    # Load the data
    fids = load_metric_data('fid50k')
    mae_fid_data = {
        key: fids[key] for key in fids.keys() if key in ['MAE-XL', 'MAE-L']
    }
    dit_fid_data = {
        key: fids[key] for key in fids.keys() if key in 
        ['DiT-XL', 'DiT-L']
    }
    
    # Plot the data
    fid_plot_func(dit_fid_data, mae_fid_data)
    print('saved plot to ./visuals/fid/training_steps_plot.png')
if __name__ == "__main__":
    main()
import matplotlib.pyplot as plt
import numpy as np
from math import log10
from dit_flops import get_dit_flops
dit_gflops = {
    (model_config, patch_size): get_dit_flops(model_config, patch_size)
    for model_config in ["XL", "L"]
    for patch_size in [2]
}
min_dit_gflops = log10(min(dit_gflops.values()))
max_dit_gflops = log10(max(dit_gflops.values()))
    
def marker_size(config, patch_size, min_val=4, max_val=12):
    config = "S" if config == "s" else config
    config = "XL" if config == "H" else config
    key = (config, patch_size)
    if key not in dit_gflops:
        raise NotImplementedError(f"GFLOPs not implemented for {key}")
    return min_val + (log10(dit_gflops[key]) - min_dit_gflops) / (max_dit_gflops - min_dit_gflops) * (max_val - min_val)

def plot_and_save(fit_ode, fit_sde, dit_ode, dit_sde, config, patch_size, plot_sde=False):
    DARK_MODE = False
    plt.clf()
    # make plot high resolution:
    plt.rcParams['figure.dpi'] = 400
    # plt.rcParams["figure.figsize"] = (10,8)
    plt.rcParams['text.usetex'] = True
    if DARK_MODE:
        # make background TRANSPARENT, not white:
        plt.rcParams['figure.facecolor'] = 'black'
        plt.rcParams['savefig.facecolor'] = 'black'
        plt.rcParams['savefig.edgecolor'] = 'black'
        # make text white:
        plt.rcParams['text.color'] = 'white'
        plt.rcParams['axes.labelcolor'] = 'white'
        plt.rcParams['xtick.color'] = 'white'
        plt.rcParams['ytick.color'] = 'white'
        plt.rcParams['axes.edgecolor'] = 'white'
    plt.rc('font', family='Helvetica')
    # import seaborn as sns

    # pal = sns.color_palette("rocket")
    pal = ["#AEC6FE", "#86AAFE", "#5D8DFD", "#B3CDA2", "#9EBF88", "#88B06D", "#F9CB8A", "#F7BA64", "#F5A83D", "#FBA09D", "#F86762", "#F5433D"]
    pal = np.flip(np.asarray(pal).reshape(4, 3), axis=1)
    print(pal[3, 0].item())
    pal = {
        "DiT": ["#5d8dfd", "#88B06D"],#, "#6d98fd", "#7da4fd", "#8eaffe"],
        "FiT": ["#f5433d", "#F5A83D"]#"#f65650", "#f76964", "#f87b77"]
    }

    ax = plt.gca()
    # Add light grey grid lines to subplots and remove x and y axis ticksa:
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='both', length=0)
    if DARK_MODE:
        ax.set_facecolor('black')
        ax.patch.set_facecolor('black')
    
    fit_ode_steps, fit_ode_fid = fit_ode[config][patch_size]
    fit_sde_steps, fit_sde_fid = fit_sde[config][patch_size]
    fit_ode_steps, fit_ode_fid = fit_ode_steps[1:], fit_ode_fid[1:]
    fit_sde_steps, fit_sde_fid = fit_sde_steps[1:], fit_sde_fid[1:]

    dit_ode_steps, dit_ode_fid = dit_ode[config][patch_size]
    dit_sde_steps, dit_sde_fid = dit_sde[config][patch_size]
    dit_ode_steps, dit_ode_fid = dit_ode_steps[1:], dit_ode_fid[1:]
    dit_sde_steps, dit_sde_fid = dit_sde_steps[1:], dit_sde_fid[1:]

    # fit_train_steps, fit_fid = fit_data[config][patch_size]
    # dit_train_steps, dit_fid = dit_data[config][patch_size]
    # fit_train_steps, fit_fid = fit_train_steps[1:], fit_fid[1:]
    # dit_train_steps, dit_fid = dit_train_steps[1:], dit_fid[1:]
    
    # if config in ["s", "S"]:
    ### SDE fine tune
    if config == "B":
        dit_sde_steps, dit_sde_fid = dit_sde_steps[3:-1], dit_sde_fid[3:-1]
    elif config in ["XL", "H"]:
        dit_sde_steps, dit_sde_fid = dit_sde_steps[4:], dit_sde_fid[4:]
    else:
        dit_sde_steps, dit_sde_fid = dit_sde_steps[3:], dit_sde_fid[3:]

    ### ODE fine tune
    if config in ["s", "S"]:
        dit_ode_steps, dit_ode_fid = dit_ode_steps[:-1], dit_ode_fid[:-1]
    elif config in ["XL", "H"]:
        dit_ode_steps, dit_ode_fid = dit_ode_steps[1:], dit_ode_fid[1:]
    
    lower = min(fit_sde_fid)
    upper = max(dit_ode_fid)
    if config in ["XL", "H"]:
        ax.set_ylim(lower - 2.5, upper + 1)
    else:
        ax.set_ylim(lower - 5, upper + 2)

    suffix = "SDE" if plot_sde else "ODE"

    if plot_sde:
        ax.plot(
            fit_sde_steps, 
            fit_sde_fid, 
            '-',
            label=f"SiT-{config}",  
            color=pal["FiT"],
            marker="o", 
            linewidth=marker_size(config, patch_size) / 4,
            markersize=marker_size(config, patch_size),
        )
        ax.plot(
            dit_sde_steps, 
            dit_sde_fid, 
            '-',
            label=f"DiT-{config}",  
            color=pal["DiT"],
            marker="o", 
            linewidth=marker_size(config, patch_size) / 4,
            markersize=marker_size(config, patch_size),
        )

        ax.plot(
            fit_ode_steps, 
            fit_ode_fid, 
            '-',
            label=str(),  
            color=pal["FiT"],
            marker="o", 
            linewidth=marker_size(config, patch_size) / 4,
            markersize=marker_size(config, patch_size),
            alpha=0.05
        )
        ax.plot(
            dit_ode_steps, 
            dit_ode_fid, 
            '-',
            label=str(),  
            color=pal["DiT"],
            marker="o", 
            linewidth=marker_size(config, patch_size) / 4,
            markersize=marker_size(config, patch_size),
            alpha=0.05
        )
    else:
        ax.plot(
            fit_ode_steps, 
            fit_ode_fid, 
            '-',
            label=f"SiT-{config} (ODE)",  
            color=pal["FiT"][1],
            marker="o", 
            linewidth=marker_size(config, patch_size) / 4,
            markersize=marker_size(config, patch_size),
        )
        ax.plot(
            dit_ode_steps, 
            dit_ode_fid, 
            '-',
            label=f"DiT-{config} (DDIM)",  
            color=pal["DiT"][1],
            marker="o", 
            linewidth=marker_size(config, patch_size) / 4,
            markersize=marker_size(config, patch_size),
        )
    # Label bottom x-axes as "Training Steps" and left y-axes as "FID-50K":
    x_label = f"Training Steps"
    ax.set_xlabel(x_label, fontsize=16)
    # Set x-ticks to use "K" instead of 1e3:
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(round(x/1e3, 1))) + "K"))
    # increase x-tick label size:
    ax.tick_params(axis='x', labelsize=16)
    y_label = "FID-50K"
    ax.set_ylabel(y_label, fontsize=16)
    # increase y-tick label size:
    ax.tick_params(axis='y', labelsize=16)
    # add legend:
    ax.legend(fontsize=16, loc="upper right") if not DARK_MODE else ax.legend(fontsize=16, loc="upper right", facecolor="black", edgecolor="black")

    plt.tight_layout()
    plt.savefig(f"outputs/SiT-{config}_ODE_samples.pdf")
    
# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .simple_diffusion import SimpleDiffusion

def create_diffusion(
    timestep_respacing,
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
    input_base_dimension_ratio: float = 1.0,
    use_simple_diffusion: bool = False,
    use_loss_weighting: bool = False,
) -> gd.GaussianDiffusion:
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    if use_simple_diffusion:
        diffusion = SimpleDiffusion(
            size_ratio=input_base_dimension_ratio,
            schedule=gd.ScheduleType.COSINE,
            pred_term=gd.ModelMeanType.VELOCITY,
            loss_type=gd.LossType.WEIGHTED_MSE if use_loss_weighting else gd.LossType.MSE,
            diffusion_steps=diffusion_steps,
            used_timesteps = space_timesteps(diffusion_steps, timestep_respacing)
        )
        return diffusion
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        input_base_dimension_ratio=input_base_dimension_ratio,
        # rescale_timesteps=rescale_timesteps,
    )

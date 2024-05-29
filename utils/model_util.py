
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from diffusion.gaussian_diffusion_simple import GaussianDiffusionSimple

def create_gaussian_diffusion_simple(args, model, modeltype, clip_model):
    steps = 1000
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule('cosine', steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return GaussianDiffusionSimple(args, model, modeltype, clip_model, betas)


def create_gaussian_diffusion(args, model, modeltype, clip_model):
    steps = 1000
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule('cosine', steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    # return GaussianDiffusionSimple(args, model, modeltype, clip_model, betas)

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=gd.ModelVarType.FIXED_SMALL,
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=0.0,
        lambda_rcxyz=0.0,
        lambda_fc=0.0,
    )
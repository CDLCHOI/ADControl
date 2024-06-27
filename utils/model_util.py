
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from diffusion.gaussian_diffusion_simple import GaussianDiffusionSimple
from utils.mask_utils import TextCLIP, calc_loss_xyz, vis_motion
import clip
import torch
import logging
from os.path import join as pjoin
from sys import stdout

def get_clip_model():
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
    clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_model = TextCLIP(clip_model)
    return clip_model

def create_gaussian_diffusion_simple(args, model, modeltype, clip_model=get_clip_model()):
    steps = 1000
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.

    betas = gd.get_named_beta_schedule('cosine', steps, scale_beta)

    if not timestep_respacing:
        timestep_respacing = [steps]

    return GaussianDiffusionSimple(args, model, modeltype, clip_model, betas)

def sample_ADControl(diffusion_root, diffusion, args, model_kwargs, vis=False):
    """sample for both stage1 and stage2. Only suppert HumanML3D dataset
    """
    # stage1
    pred_ric = diffusion_root.p_sample_loop(partial_emb=None, model_kwargs=model_kwargs,batch_size=args.batch_size)
    if vis:
        vis_motion(pred_ric[0], model_kwargs['gt_ric'][0], dataset=args.dataset_name, save_path=f'./output/testsample/stage1.html')

    if args.normalize_traj:
        traj = model_kwargs['traj'] * diffusion_root.raw_std + diffusion_root.raw_mean
    loss_xyz = calc_loss_xyz(pred_ric, traj, model_kwargs['traj_mask']) # 仅约束控制的关节误差
    
    # stage2
    sample = []
    for j in range(args.stage2_repeat_times):
        partial_emb = torch.zeros_like(model_kwargs['gt_motion'], device=model_kwargs['gt_motion'].device)
        partial_emb[..., :67] = pred_ric  
        pred_motion = diffusion.p_sample_loop(partial_emb, with_control=True, model_kwargs=model_kwargs, batch_size=args.batch_size) 
        if vis:
            vis_motion(pred_motion[0], model_kwargs['gt_motion'][0], dataset=args.dataset_name, save_path=f'./output/testsample/stage2_{j+1}.html')
        sample.append(pred_motion)
    
    return sample, loss_xyz




def get_semanticboost_args(args):

    clip_version = 'ViT-B/32'
    args.arch = "llama_decoder_rope"
    # cond_mode = 'no_cond'
    cond_mode = "text"
     
    activation = "swiglu"

    if args.dataset_name == 't2m':
        njoints = 263
        nfeats = 1
        dataset = "humanml"
    elif args.dataset_name == 'kit':
        njoints = 251
        nfeats = 1
        dataset = "kit"

    return {'njoints': njoints, 'nfeats': nfeats, 'latent_dim': 512, 'ff_size': 1024, 'num_layers': 8, 'num_heads': 4,
            'dropout': 0.1, 'activation': activation, 'cond_mode': cond_mode, 'cond_mask_prob': 0.1, 'arch': args.arch,
            'clip_version': clip_version, 'dataset': dataset, "local":False, "encode_full":2, "txt_tokens":2,
            "num_frames":196, "conv_bias":True, "conv_activate":"relu", 
            "conv_norm":"layernorm"}

def initial_optim(lr, weight_decay, net, optimizer) : 
    
    if optimizer == 'adamw' : 
        optimizer_adam_family = torch.optim.AdamW
    elif optimizer == 'adam' : 
        optimizer_adam_family = torch.optim.Adam
    
    optimizer = optimizer_adam_family(net.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=weight_decay)
        
        
    return optimizer

def get_logger(out_dir, file_path=None):
    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    if file_path == None:
        file_path = pjoin(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger



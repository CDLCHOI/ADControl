import options.option_transformer as option_trans
import os 
args = option_trans.get_args_parser()
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)
from utils.fixseed import fixseed
# fixseed(99824)
# fixseed(10004)
# fixseed(4332)
# fixseed(1111) # 站着挥手臂
import torch
import numpy as np
import sys
# from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
import clip

from dataset import dataset_control

from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from recoverML3D import get_ML3D_emb
import matplotlib.pyplot as plt
from utils.mask_utils import calc_grad_scale, TextCLIP, calc_loss_xyz, load_ckpt
from utils.metrics import evaluate_control
from utils.model_util import create_gaussian_diffusion_simple, get_logger
from utils.text_control_example import collate_all


args.stage2_repeat_times = 1
if sys.gettrace():
    args.control_joint = 21
    args.density = 100

args.resume_root = 'output/0518_omni67_multi_partxyz/net_last.pth'; args.roottype = 'omni67'; outname = 'omni67' ; args.normalize_traj=True 
# args.resume_root = 'output/0520_omni67_noxyzrootloss/net_last.pth'; args.roottype = 'omni67'; outname = 'omni67' ; args.normalize_traj=True 


if sys.gettrace():
    log_file = f'output/ttt/1.log'
    if os.path.exists(log_file):
        os.remove(log_file)
else:
    log_file = f'{os.path.dirname(args.resume_root)}/joint_{args.control_joint}_density_{args.density}.log'
    # log_file = f'{os.path.dirname(args.resume_root)}/joint_{args.control_joint}_density_{args.density}_controlonce.log'
logger = get_logger('', file_path=log_file)
logger.info(f'log_file = {log_file}')
logger.info(f'args.resume_root = {args.resume_root}')
logger.info(f'control joint = {args.control_joint}, density = {args.density}')

args.resume_trans = 'output/0509_diffmae_stage2_2_E8D0_multicontrol_pretrain/net_last.pth'; args.modeltype = 'diffmae_stage2_2' 
# args.resume_trans = 'output/0519_semboost_noxyzloss/net_last.pth'; args.modeltype = 'semboost'

if __name__ == '__main__':

    mean = np.load('dataset/HumanML3D/Mean.npy')[None, ...] # dataset/HumanML3D/Mean.npy
    std = np.load('dataset/HumanML3D/Std.npy')[None, ...]
    humanml_mean = torch.from_numpy(mean)[None, ...].cuda()
    humanml_std = torch.from_numpy(std)[None, ...].cuda()

    raw_mean = torch.from_numpy(np.load('dataset/humanml_spatial_norm/Mean_raw.npy')).cuda()[None, None, ...].view(1,1,22,3) 
    raw_std = torch.from_numpy(np.load('dataset/humanml_spatial_norm/Std_raw.npy')).cuda()[None, None, ...].view(1,1,22,3)

    ##### ---- CLIP ---- #####
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
    clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_model = TextCLIP(clip_model)

    # 根节点网络
    if args.roottype == 'omni67':
        from models.omni67 import CMDM
        net_root = CMDM(args, args.roottype)
    diffusion_root = create_gaussian_diffusion_simple(args, net_root, args.roottype, clip_model)
    load_ckpt(net_root, args.resume_root, key='trans')
    net_root.eval()
    net_root.cuda()

    # 2阶段网络暂时与控制精度无关，暂时注释
    # if args.modeltype == 'diffmae_stage2_2':
    #     from models.diffmae_2 import DiffMAE2
    #     net = DiffMAE2(dataset=args.dataset_name, args=args, num_layers_E=8, num_layers_D=0)

    # diffusion = create_gaussian_diffusion_simple(args, net, args.modeltype, clip_model)

    # 读取权重
    # ckpt = torch.load(args.resume_trans, map_location='cpu')
    # if 'module' in list(ckpt['trans'].keys())[0]:
    #     new_ckpt = {}
    #     for k, v in ckpt['trans'].items():
    #         new_k = k.replace('module.', '') if 'module' in k else k
    #         new_ckpt[new_k] = v
    #     net.load_state_dict(new_ckpt, strict=True)
    # else:
    #     net.load_state_dict(ckpt['trans'], strict=True)
    # net.eval()
    # net.cuda()

    eval_batch = 128
    test_loader = dataset_control.DataLoader(batch_size=eval_batch, args=args, mode='eval', split='test', shuffle=False, num_workers=0, drop_last=True)
    logger.info(f'eval_batch = {eval_batch}')

    evaluate_control(test_loader, diffusion_root, humanml_mean, humanml_std, args, logger, batch_size=eval_batch, diffusion=None)
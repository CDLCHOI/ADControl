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
from torch.distributions import Categorical
import clip
import json
import models.vqvae as vqvae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
import models.t2m_trans as trans

from dataset import dataset_control
from models.transformer_ED import Transformer_ED

from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from exit.utils import get_model, visualize_2motions, generate_src_mask, init_save_folder, uniform, cosine_schedule, gumbel_sample
from utils.motion_process import recover_from_ric, recover_root_rot_pos
from einops import rearrange, repeat
import torch.nn.functional as F
import shutil
from recoverML3D import get_ML3D_emb
import utils.losses as losses 
import matplotlib.pyplot as plt
from utils.mask_utils import gradients, calc_grad_scale, random_mask_motion2, TextCLIP, complete_mask, calc_loss_xyz, create_trajmask263
from utils.metrics import evaluate_control, evaluate_control_diffmae
from utils.model_util import create_gaussian_diffusion, create_gaussian_diffusion_simple
from utils.text_control_example import collate_all

# args = option_trans.get_args_parser()

args.roottype = 'root'
args.exp_name = 'debug_model'
args.no_random_mask = True
args.print_iter = 1
args.overwrite = True
args.total_iter_ROOT = 20
args.total_iter_ED = 1 
args.modeltype = 'ED'
args.dense_control = False
args.stage2_repeat_times = 1
args.use_stage1 = False
if sys.gettrace():
    args.control_joint = 21
    args.density = 100

# args.resume_root = 'output/trans_0407_root_maskmin_0_5/net_last.pth'
# args.resume_root = 'output/trans_0410_E3D1_root/net_last.pth'
# args.resume_root = 'output/trans_0413_root/net_last.pth'
# args.resume_root = 'output/trans_0413_root_rootdistloss/net_last.pth'
# args.resume_root = 'output/trans_0415_root_rootdistloss/net_last.pth' # 加入对根会随机100%mask
# args.resume_root = 'output/trans_0419_root_masknoise/net_last.pth'; args.mask_noise = True # 加入对mask掉的值加高斯噪声
# args.resume_root = 'output/trans_0420_root_masknoiseall/net_last.pth'; args.root_mask_noise_all = True 

# args.resume_root = 'output/trans_0420_root2/net_last.pth'; args.roottype = 'root2' # 网络预测输出改为4个root+67个ric
# args.resume_root = 'output/trans_0420_root2_noise/net_last.pth'; args.roottype = 'root2'; args.mask_noise=True
# args.resume_root = 'output/trans_0420_root2_noiseall/net_last.pth'; args.roottype = 'root2'; args.root_mask_noise_all=True

# args.resume_root = 'output/0508_diffmae_root67_E8D0_add/net_last.pth'; args.roottype = 'diffmae_root67'; args.num_layers_E=8; args.num_layers_D=0; args.add_traj_feat=True; outname = 'diffmaeroot67'
# args.resume_root = 'output/0508_omni67/net_last.pth'; args.roottype = 'omni67'; outname = 'omni67' # 用omnicontrol搭的根网络  √！！
# args.resume_root = 'output/0513_omni67_normtraj_multicontrol/net_last.pth'; args.roottype = 'omni67'; outname = 'omni67' ; args.normalize_traj=True # 归一化轨迹再输入
# args.resume_root = 'output/0514_omni67_normtraj_multicontrol/net_last.pth'; args.roottype = 'omni67'; outname = 'omni67' ; args.normalize_traj=True # 归一化轨迹再输入
# args.resume_root = 'output/0517_omni67/net_last.pth'; args.roottype = 'omni67'; outname = 'omni67' ; args.normalize_traj=True 
# args.resume_root = 'output/0517_omni67_noxyzrootloss/net_last.pth'; args.roottype = 'omni67'; outname = 'omni67' ; args.normalize_traj=True 
# args.resume_root = 'output/0517_omni67_xyzpart/net_last.pth'; args.roottype = 'omni67'; outname = 'omni67' ; args.normalize_traj=True 
args.resume_root = 'output/0518_omni67_multi_partxyz/net_last.pth'; args.roottype = 'omni67'; outname = 'omni67' ; args.normalize_traj=True 
# args.resume_root = 'output/0520_omni67_noxyzrootloss/net_last.pth'; args.roottype = 'omni67'; outname = 'omni67' ; args.normalize_traj=True 


if sys.gettrace():
    log_file = f'output/ttt/1.log'
    if os.path.exists(log_file):
        os.remove(log_file)
else:
    # log_file = f'{os.path.dirname(args.resume_root)}/joint_{args.control_joint}_density_{args.density}.log'
    log_file = f'{os.path.dirname(args.resume_root)}/joint_{args.control_joint}_density_{args.density}_controlonce.log'
logger = utils_model.get_logger('', file_path=log_file)
logger.info(f'log_file = {log_file}')
logger.info(f'args.resume_root = {args.resume_root}')
logger.info(f'control joint = {args.control_joint}, density = {args.density}')


# args.resume_trans = 'output/trans_0410_E3D1_allxyz_nomaskvel_pretrain/net_last.pth'
# args.resume_trans = 'output/trans_0413_ED/net_last.pth'  # 对照组
# args.resume_trans = 'output/trans_0413_ED_partxyz_maskroot/net_last.pth' # mask_root
# args.resume_trans = 'output/trans_0413_ED_partxyz_maskroot_norootcontrol/net_last.pth' # mask_root
# args.resume_trans = 'output/trans_0413_ED_partxyz_rootdistloss/net_last.pth' # root_dist_loss
# args.resume_trans = 'output/trans_0413_ED_partxyz_maskroot_rootdistloss/net_last.pth' # 都加
# args.resume_trans = 'output/trans_0415_ED_partxyz_maskroot_rootdistloss/net_last.pth' # 换了random_mask_motion2训练
# args.resume_trans = 'output/0424_trans_ED_E6D2/net_last.pth'; args.num_layers_E = 6; args.num_layers_D = 2  # 挺差
# args.resume_trans = 'output/0425_trans_ED_E6D2/net_last.pth'; args.num_layers_E = 6; args.num_layers_D = 2  # 挺差
# args.resume_trans = 'output/0425_trans_ED_E6D1/net_last.pth'; args.num_layer s_E = 6; args.num_layers_D = 1  # 挺差
# args.resume_trans = 'output/trans_0425_ED_masknoise/net_last.pth'; args.mask_noise=True

# args.resume_trans = 'output/0427_diffmae_stage2_pretrain/net_last.pth'; args.modeltype = 'diffmae_stage2' # diffmae

# args.resume_trans = 'output/0429_diffmae_stage2_2_E8D0/net_last.pth'; args.modeltype = 'diffmae_stage2_2' 
# args.resume_trans = 'output/0429_diffmae_stage2_2_E8D0_nomask/net_last.pth'; args.modeltype = 'diffmae_stage2_2' 
# args.resume_trans = 'output/0429_diffmae_stage2_2_E8D0_rootloss/net_last.pth'; args.modeltype = 'diffmae_stage2_2' 

# args.resume_trans = 'output/0430_diffmae_stage2_2_E8D0/net_last.pth'; args.modeltype = 'diffmae_stage2_2' 
# args.resume_trans = 'output/0430_diffmae_stage2_2_E8D0_noroot/net_last.pth'; args.modeltype = 'diffmae_stage2_2' 
# args.resume_trans = 'output/0430_diffmae_stage2_2_E8D0_nomask/net_last.pth'; args.modeltype = 'diffmae_stage2_2' # MDM
args.resume_trans = 'output/0509_diffmae_stage2_2_E8D0_multicontrol_pretrain/net_last.pth'; args.modeltype = 'diffmae_stage2_2' 



args.out_dir = './output/testsample/'
sample_max_steps = args.total_iter + 1e-8


def root_dist_loss(gt_root, pred_root, real_mask):
    gt_rot, gt_pos = recover_root_rot_pos(gt_root)
    pred_rot, pred_pos = recover_root_rot_pos(pred_root)
    loss_rotate = F.l1_loss(gt_rot[real_mask], pred_rot[real_mask], reduction='mean')
    loss_position = F.l1_loss(gt_pos[real_mask], pred_pos[real_mask], reduction='mean')
    return loss_rotate, loss_position, gt_pos, pred_pos

if __name__ == '__main__':
    Loss = losses.ReConsLoss('l1_smooth', 22)

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
    if args.roottype == 'diffmae_root67':
        from models.diffmae_root67 import DiffMAERoot67
        net_root = DiffMAERoot67(dataset=args.dataname, args=args, num_features=4+63, num_layers_E=args.num_layers_E, num_layers_D=args.num_layers_D)
    elif args.roottype == 'omni67':
        from models.omni67 import CMDM
        net_root = CMDM(args, args.roottype)
    diffusion_root = create_gaussian_diffusion_simple(args, net_root, args.roottype, clip_model)

    ckpt = torch.load(args.resume_root, map_location='cpu')
    if 'module' in list(ckpt['trans'].keys())[0]:
        new_ckpt = {}
        for k, v in ckpt['trans'].items():
            new_k = k.replace('module.', '') if 'module' in k else k
            new_ckpt[new_k] = v
        net_root.load_state_dict(new_ckpt, strict=True)
    else:
        net_root.load_state_dict(ckpt['trans'], strict=True)
    net_root.eval()
    net_root.cuda()

    # 2阶段网络
    if args.modeltype == 'diffmae_stage2':
        from models.diffmae import DiffMAE
        net = DiffMAE(dataset=args.dataname, args=args, num_layers_E=args.num_layers_E, num_layers_D=args.num_layers_D)

    elif args.modeltype == 'diffmae_stage2_2':
        from models.diffmae_2 import DiffMAE2
        net = DiffMAE2(dataset=args.dataname, args=args, num_layers_E=8, num_layers_D=0)

    diffusion = create_gaussian_diffusion_simple(args, net, args.modeltype, clip_model)

    # 读取权重
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    if 'module' in list(ckpt['trans'].keys())[0]:
        new_ckpt = {}
        for k, v in ckpt['trans'].items():
            new_k = k.replace('module.', '') if 'module' in k else k
            new_ckpt[new_k] = v
        net.load_state_dict(new_ckpt, strict=True)
    else:
        net.load_state_dict(ckpt['trans'], strict=True)
    net.eval()
    net.cuda()

    eval_batch = 128
    test_loader = dataset_control.DataLoader(batch_size=eval_batch, args=args, mode='eval', split='test', shuffle=False, num_workers=0, drop_last=True)
    logger.info(f'eval_batch = {eval_batch}')

    evaluate_control_diffmae(test_loader, diffusion_root, humanml_mean, humanml_std, args, logger, batch_size=eval_batch, diffusion=diffusion)
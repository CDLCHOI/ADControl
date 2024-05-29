import options.option_transformer as option_trans
import os 
args = option_trans.get_args_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
# os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)
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
# from utils.inference import interface
from utils.metrics import evaluate_control, evaluate_control_diffmae
from utils.model_util import create_gaussian_diffusion, create_gaussian_diffusion_simple
from utils.text_control_example import collate_all

from utils.rotation2xyz import Rotation2xyz

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
args.resume_root = 'output/0518_omni67_multi_partxyz/net_last.pth'; args.roottype = 'omni67'; outname = 'omni67' ; args.normalize_traj=True 

if sys.gettrace():
    log_file = f'output/ttt/1.log'
    if os.path.exists(log_file):
        os.remove(log_file)
else:
    log_file = f'{os.path.dirname(args.resume_root)}/joint_{args.control_joint}_density_{args.density}.log'
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
# args.resume_trans = 'output/0519_semboost_noxyzloss/net_last.pth'; args.modeltype = 'semboost'



args.out_dir = './output/testsample/'
sample_max_steps = args.total_iter + 1e-8

def draw_xz_map(gt, pred):
    x = gt[0,:,0].detach().cpu().numpy()
    z = gt[0,:,2].detach().cpu().numpy()
    plt.scatter(x, z, c='r')
    x = pred[0,:,0].detach().cpu().numpy()
    z = pred[0,:,2].detach().cpu().numpy()
    plt.scatter(x[::10], z[::10])
    plt.savefig(f'{args.out_dir}/xz.png')
    a = 1

def draw_xyz_map(gt, pred):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = gt[0,:,0].detach().cpu().numpy()
    y = gt[0,:,1].detach().cpu().numpy()
    z = gt[0,:,2].detach().cpu().numpy()
    ax.scatter(x, y, z, c='r')

    x = pred[0,:,0].detach().cpu().numpy()
    y = pred[0,:,1].detach().cpu().numpy()
    z = pred[0,:,2].detach().cpu().numpy()
    ax.scatter(x, y, z, c='r')
    
    # 设置坐标轴标签
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

def root_dist_loss(gt_root, pred_root, real_mask):
    gt_rot, gt_pos = recover_root_rot_pos(gt_root)
    pred_rot, pred_pos = recover_root_rot_pos(pred_root)
    loss_rotate = F.l1_loss(gt_rot[real_mask], pred_rot[real_mask], reduction='mean')
    loss_position = F.l1_loss(gt_pos[real_mask], pred_pos[real_mask], reduction='mean')
    return loss_rotate, loss_position, gt_pos, pred_pos

def random_mask_motion(motion, traj_mask_263, ratio, real_length = None):
    '''
    保留motion中与控制关节点相关的值; 将非控制关节点部分的随机mask置0
    motion: (b,196.263)
    traj_mask_263: (b,196,263,)
    real_length: (b)
    '''
    B, L, num_features = motion.shape
    ### temporal mask  TODO: 时间mask还不严谨，应该取真是长度来算mask的个数，但是创建mask的时候，只能在有效长度内执行mask，有效长度外的必须为False
    if real_length == None:
        real_length = torch.tensor([L]).float()

    num_token_masked = (real_length * ratio).round().clamp(min=1) .cuda()
    # 下面两行目的就是按照生成的motion mask百分比，生成随机的mask。实现思路是：生成一串随机数，通过排序索引，然后将索引小于要被mask数量的位置就变成True，以此实现随机mask生成
    batch_randperm = torch.rand((L)).argsort(dim=-1).cuda()
    temporal_mask = ~(batch_randperm < num_token_masked.unsqueeze(-1)) # 要被mask的位置是False  (L)

    ### spatial mask
    num_token_masked = (torch.tensor([num_features]).to(ratio.device) * ratio).round().clamp(min=1).cuda()
    batch_randperm = torch.rand((num_features)).argsort(dim=-1).cuda()
    spatial_mask = ~(batch_randperm < num_token_masked.unsqueeze(-1)) # 要被mask的位置是False  (263)

    mask = (temporal_mask[...,None] & spatial_mask[:,None]) | traj_mask_263
    masked_motion = motion * mask
    return masked_motion.float()

def random_mask_root(gt_root, real_length, ratio):
    '''
    gt_root: (b,196,4)
    real_length: (b,
    ratio:  
    return : (b,196,4)
    '''
    B, L, num_features = gt_root.shape
    
    num_token_masked = (real_length * ratio).round().clamp(min=1)  # (b,)
    batch_randperm = torch.rand((B, L)).argsort(dim=-1).cuda()
    temporal_mask = ~(batch_randperm < num_token_masked.unsqueeze(-1)) # 要被mask的位置是False  (b,L)
    
    real_mask = generate_src_mask(L, real_length)
    mask = (temporal_mask & real_mask)[..., None] # 与real_mask做与运算，将有效长度以外的必定mask
    masked_root = gt_root * mask
    return masked_root.float()

def random_mask_root_prob(gt_root,real_length, ratio):
    '''random mask batch root data in random ratio in 0.5->1
    gt_root: (b,196,4)
    real_length: (b,)
    return : (b,196,4)
    '''
    B, L, num_features = gt_root.shape
    thr = torch.zeros(B,1).uniform_(ratio,ratio).clamp(max=1)
    batch_mask = (torch.zeros(B,L).uniform_(0,1)>thr).to(real_length.device) # 实际长度的mask
    length_mask = (torch.arange(L).expand(B,L).to(real_length.device)) < real_length[...,None] # 有效长度的mask 
    batch_mask = batch_mask & length_mask
    masked_root = gt_root*batch_mask[...,None]
    if args.mask_noise:
        noise = torch.randn_like(masked_root, device=masked_root.device)
        masked_root += noise * ~batch_mask[...,None]
    return masked_root.float()

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
    if args.roottype == 'root':
        from models.transformer_root import Transformer_Root
        net_root = Transformer_Root(args=args, num_features=4, num_layers_E=3, num_layers_D=1)
    elif args.roottype == 'root2':
        from models.transformer_root2 import Transformer_Root2
        net_root = Transformer_Root2(args=args, num_features=4+63, num_layers_E=3, num_layers_D=1)
    elif args.roottype == 'diffmae_root67':
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
    if args.modeltype == 'ED':
        net = Transformer_ED(dataset = args.dataname, args = args, num_layers_E=args.num_layers_E, num_layers_D=args.num_layers_D)
    elif args.modeltype == 'diffmae_stage2':
        from models.diffmae import DiffMAE
        net = DiffMAE(dataset=args.dataname, args=args, num_layers_E=args.num_layers_E, num_layers_D=args.num_layers_D)
    elif args.modeltype == 'diffmae_stage2_2':
        from models.diffmae_2 import DiffMAE2
        net = DiffMAE2(dataset=args.dataname, args=args, num_layers_E=8, num_layers_D=0)
    elif args.modeltype == 'semboost':
        from models.semboost import MDM
        from utils.semboost_utils import get_semboost_args
        net = MDM(**get_semboost_args(args))

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

    #从dataset中随机抽取，取其text与控制joint的263 dim数据
    train_loader = dataset_control.DataLoader(batch_size=1, args=args, mode='eval', shuffle=True,)
    # train_loader_iter = dataset_control.cycle(train_loader)
    eval_batch = 128
    # val_loader = dataset_control.DataLoader(batch_size=eval_batch, args=args, mode='eval', split='val', shuffle=True, num_workers=0)
    # val_loader_iter = dataset_control.cycle(val_loader)
    test_loader = dataset_control.DataLoader(batch_size=eval_batch, args=args, mode='eval', split='test', shuffle=False, num_workers=0, drop_last=True)
    logger.info(f'eval_batch = {eval_batch}')
    # evaluate_control(val_loader, net, humanml_mean, humanml_std, total_iter=1)

    # evaluate_control_diffmae(test_loader, diffusion_root, humanml_mean, humanml_std, args, logger, batch_size=eval_batch)

    # diffusion.p_sample_single(train_loader_iter)
    
    rot2xyz = Rotation2xyz(device='cuda:0')
    for i, batch in enumerate(train_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, gt_motion, real_length, txt_tokens, traj, traj_mask_263, traj_mask = batch
        b, max_length, num_features = gt_motion.shape
        gt_motion = gt_motion.cuda()
        real_length = real_length.cuda()
        traj = traj.cuda()
        traj_mask = traj_mask.cuda()
        traj_mask_with_root = traj_mask.clone()
        traj_mask_with_root[:, :real_length, 0, :] = True # 这个操作只支持batch为1
        traj_mask_263 = traj_mask_263.cuda()
        traj_mask_263_with_root = traj_mask_263.clone()
        traj_mask_263_with_root[..., :4] = True
        real_mask = generate_src_mask(max_length, real_length) # (b,196)
        gt_ric = gt_motion[..., :67]
        print(clip_text)
        # if not 'walk' in clip_text[0]:
        #     continue
        print('real length = ', real_length)

        #encode text
        text = clip.tokenize(clip_text, truncate=True).cuda()        
        text_emb, word_emb = clip_model(text) # (b,512) 

        condition = {}
        condition['traj'] = traj.clone()
        condition['text_emb'] = text_emb
        condition['word_emb'] = word_emb
        condition['traj_mask'] = traj_mask
        condition['traj_mask_263'] = traj_mask_263
        condition['gt_motion'] = gt_motion
        condition['real_mask'] = real_mask
        condition['clip_text'] = clip_text

        #### 6D旋转可视化
        # 
        # rot = gt_motion[:,:real_length[0], 67:193].reshape(b,real_length[0],21,6)   # (1,2)
        # xyz = rot2xyz(rot, mask=None,
        #                 pose_rep='rot6d', translation=False, glob=True,
        #                 # jointstype='vertices',
        #                 jointstype='smpl',  # for joint locations
        #                 vertstrans=True)

        if args.normalize_traj:
            traj = traj * raw_std + raw_mean
        # # 采样根节点轨迹
        print('=== root sample')
        if args.roottype == 'diffmae_root67':
            pred_ric = diffusion_root.p_sample_loop(partial_emb=None, model_kwargs=condition)
        elif args.roottype == 'omni67': 
            pred_ric = diffusion_root.p_sample_loop(partial_emb=None, model_kwargs=condition)

        args.use_stage1 = True
        loss_rotate_global, root_position_global, gt_root_pos, pred_root_pos = root_dist_loss(
                    gt_ric[..., :4] * humanml_std[..., :4] + humanml_mean[..., :4], 
                    pred_ric[..., :4] * humanml_std[..., :4] + humanml_mean[..., :4], 
                    real_mask)
        control_id = traj_mask[0].sum(0).sum(-1).nonzero()
        print('control_id = ', control_id)
        
        recon_xyz = recover_from_ric(pred_ric[..., :67] * humanml_std[..., :67] + humanml_mean[..., :67], joints_num=22) 
        loss_xyz_part = F.l1_loss(recon_xyz[traj_mask], traj[traj_mask]) # 仅约束控制轨迹
        print('loss_xyz_part = ', loss_xyz_part)
        print('root_position_global = ', root_position_global)
        draw_xz_map(gt_root_pos, pred_root_pos) # Y的人的左右
        a = 1

        if args.roottype == 'root':
            gt_root = gt_motion[..., :4]
            if args.mask_noise or args.root_mask_noise_all:
                pred_root = torch.randn_like(gt_root).to(gt_motion.device) 
            else:
                pred_root = torch.zeros(gt_root.shape).to(gt_motion.device) 
            pred_root = random_mask_root_prob(gt_root, real_length, 0.7) # 模仿训练中mask掉gt再输入
            # print('traj.sum()=',traj.sum())
            # print('word_emb.sum()=', word_emb.sum())
            # print('text_emb.sum()=', text_emb.sum())
            with torch.no_grad():
                for nb_iter in tqdm(range(1, args.total_iter_ROOT + 1), position=0, leave=True):
                    pred_root = net_root(pred_root, text_emb.unsqueeze(1), traj=traj.flatten(2,3), word_emb=word_emb) #autograd memory leak, crash with high iter number
                    timestep = torch.clamp(torch.tensor(nb_iter/(args.total_iter_ROOT)), max=1)
                    ratio = cosine_schedule(timestep)
                    if nb_iter != args.total_iter_ROOT:
                        if args.mask_noise:
                            pred_root = random_mask_root_prob(pred_root, real_length, ratio)
                        else:
                            pred_root = random_mask_root(pred_root, real_length, ratio)
            vel_mask = real_mask[..., None]
            rot_mask = real_mask[..., None].repeat(1,1,3)
            loss_vel = F.l1_loss(pred_root[..., :1][vel_mask], gt_root[..., :1][vel_mask])  # 换成L1Loss
            loss_rot = F.l1_loss(pred_root[..., 1:4][rot_mask], gt_root[..., 1:4][rot_mask])
            loss_rotate_global, root_position_global, gt_root_pos, pred_root_pos = root_dist_loss(
                            gt_root * humanml_std[..., :4] + humanml_mean[..., :4], 
                            pred_root * humanml_std[..., :4] + humanml_mean[..., :4], 
                            real_mask)
            
            print('loss_vel = ', loss_vel)
            print('loss_rot = ', loss_rot)
            print('root_position_global = ', root_position_global)
        elif args.roottype == 'root2':
            gt_ric = gt_motion[..., :67]
            ric_mask_67 = traj_mask_263[..., :67]
            root_mask_67 = torch.zeros(b, max_length, 67).cuda().bool()
            root_mask_67[..., :4] = True
            if args.mask_noise:
                pred_ric = torch.randn_like(gt_ric).to(gt_ric.device) 
            else:
                pred_ric = torch.zeros(gt_ric.shape).to(gt_ric.device) 
                
            with torch.no_grad():
                # for nb_iter in tqdm(range(1, args.total_iter_ROOT + 1), position=0, leave=True):
                for nb_iter in range(1, args.total_iter_ROOT + 1):
                    pred_ric = net_root(pred_ric, text_emb.unsqueeze(1) , traj=traj.flatten(2,3), word_emb=word_emb) # (b,196,4)
                    timestep = torch.clamp(torch.tensor(nb_iter/(args.total_iter_ROOT)), max=1)
                    # ratio = cosine_schedule(timestep)
                    # if nb_iter != args.total_iter_ROOT:
                    #     if args.mask_noise:
                    #         pred_root = random_mask_root_prob(pred_root, real_length, ratio)
                    #     else:
                    #         pred_root = random_mask_root(pred_root, real_length, ratio)
                    gt_xyz = recover_from_ric(gt_ric * humanml_std[..., :67] + humanml_mean[..., :67], joints_num=22)
                    pred_xyz = recover_from_ric(pred_ric * humanml_std[..., :67]  + humanml_mean[..., :67], joints_num=22)
                    assert torch.allclose(gt_xyz * traj_mask, traj, atol=1e-5) # 确保轨迹及mask是正确的
                    loss_xyz_part = F.l1_loss(pred_xyz[traj_mask], gt_xyz[traj_mask]) # 仅约束控制轨迹
                    print('loss_xyz_part = ', loss_xyz_part)
                    pred_root = pred_ric[..., :4]
                    if nb_iter != args.total_iter_ROOT:
                        loss, grad, pred_ric = gradients(pred_ric, gt_ric, traj, traj_mask, humanml_mean[..., :67], humanml_std[..., :67])
                        # scale = calc_grad_scale(traj_mask)
                        pred_ric = pred_ric * root_mask_67
                        # pred_ric = pred_ric * traj_mask_263[:,:,:67] 
                        ratio = cosine_schedule(timestep)
                        # pred_ric = random_mask_motion2(args, pred_ric, ric_mask_67, real_length, ratio=ratio)

            loss_rotate_global, root_position_global, gt_root_pos, pred_root_pos = root_dist_loss(
                    gt_ric[..., :4] * humanml_std[..., :4] + humanml_mean[..., :4], 
                    pred_root[..., :4] * humanml_std[..., :4] + humanml_mean[..., :4], 
                    real_mask)
        
        
        a = 1
        # 转263
        # emb, emb_mask, triplemask = get_ML3D_emb(pred_root[0], traj[0].flatten(1,2), use_hardcode_norm=True) # TODO
        # emb = emb[None, ...]

        # text, traj = collate_all()
        
        for j in range(args.stage2_repeat_times):
            # 采样动作
            print(f'=== motion sample: repeat time {j}')
            partial_emb = torch.zeros_like(gt_motion, device=gt_motion.device)
            partial_emb[..., :67] = gt_ric # 使用dataloader出来的
            if args.use_stage1:
                partial_emb[..., :67] = pred_ric # 使用1阶段网络预测出来的
            # partial_emb = emb # 使用控制轨迹和根节点转来的

            # 使用1阶段出来的pred_ric，需要把traj_mask_263中根节点的flag也置为True
            # 即2阶段强制替换只会替换拿7个值
            traj_mask_263 = complete_mask(traj_mask_263, traj_mask)

            if 'diffmae_stage2' in args.modeltype:
                # condition = {}
                # condition['traj'] = traj
                # condition['text_emb'] = text_emb
                # condition['word_emb'] = word_emb
                # condition['traj_mask'] = traj_mask
                # condition['traj_mask_263'] = traj_mask_263
                # condition['gt_motion'] = gt_motion
                # condition['real_mask'] = real_mask
                # condition['clip_text'] = clip_text
                pred_motion = diffusion.p_sample_loop(partial_emb, with_control=True, model_kwargs=condition,) # 有替换
                # pred_motion = diffusion.p_sample_loop(partial_emb, with_control=False, model_kwargs=condition,) # 无替换，即纯噪声生成和MDM一样

                # pred_motion = diffusion.p_sample_loop(
                #     net,
                #     (1,196,263),  # BUG FIX 这里本来应该是n_frames，但是会导致训练推理形状不匹配
                #     # skip_timesteps=900,
                #     # init_image=gt_motion,
                #     model_kwargs=condition,
                # )
            elif args.modeltype == 'ED':
                pred_motion = torch.zeros(gt_motion.shape).to(gt_motion.device)
                if args.mask_noise:
                    noise = torch.randn_like(pred_motion, device=pred_motion.device) # (B,L, dim)
                    pred_motion += noise 
                with torch.no_grad():
                    for nb_iter in tqdm(range(1, args.total_iter_ED + 1), position=0, leave=True):
                        pred_motion = torch.where(partial_emb == 0,pred_motion,partial_emb) # 保留预先提供的先验
                        pred_motion = net(pred_motion, text_emb, word_emb).detach() #autograd memory leak, crash with high iter number
                        timestep = torch.clamp(torch.tensor(nb_iter/(sample_max_steps)), max=1)
                        ratio = cosine_schedule(timestep)
                        if nb_iter != args.total_iter_ED:
                            pred_motion = random_mask_motion(pred_motion, traj_mask_263, ratio)

            recon_xyz = recover_from_ric(pred_motion * humanml_std + humanml_mean, joints_num=22)  # 反归一化再转全局xyz
            gt_xyz = recover_from_ric(gt_motion * humanml_std + humanml_mean, joints_num=22)
            real_mask = generate_src_mask(max_length, real_length) # (b,196)
            assert torch.allclose(gt_xyz * traj_mask, traj * traj_mask, atol=1e-5) # 确保轨迹及mask是正确的

            motion_real_mask = real_mask[..., None].repeat(1,1,num_features)
            xyz_real_mask = real_mask[..., None, None].repeat(1,1,22,3)
            # loss_rot = Loss.Loss(pred_motion[motion_real_mask], gt_motion[motion_real_mask])
            loss_xyz_part = F.l1_loss(recon_xyz[traj_mask], traj[traj_mask]) # 仅约束控制轨迹
            # print('loss_rot = ', loss_rot)
            print('loss_xyz_part = ', loss_xyz_part)
            # 计算根的loss
            # gt_root = (gt_motion * humanml_std + humanml_mean)[..., :4]
            # pred_root = (pred_motion * humanml_std + humanml_mean)[..., :4]
            # loss_rotate_global, root_position_global, gt_root_pos, pred_root_pos  = root_dist_loss(gt_root, pred_root, real_mask)
            # assert torch.allclose(gt_root_pos, gt_xyz[:,:,0,:])
            # print('root_position_global = ', root_position_global)

    
            save_name = f'./output/testsample/{outname}_{j+1}.html'
            print('save_name = ', save_name)
            visualize_2motions(pred_motion[0].detach().cpu().numpy(), std, mean, 't2m', None, motion2=gt_motion[0].detach().cpu().numpy(), save_path=save_name)
            a = 1

        
        break

    

    
    
    
    
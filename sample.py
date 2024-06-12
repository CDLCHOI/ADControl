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
from os.path import join as pjoin
import clip

from dataset import dataset_control

import warnings
warnings.filterwarnings('ignore')
import torch.nn.functional as F
from recoverML3D import get_ML3D_emb
from utils.mask_utils import calc_grad_scale, calc_loss_xyz, generate_src_mask, load_ckpt
from utils.model_util import create_gaussian_diffusion_simple, get_clip_model, sample_ADControl
from utils.text_control_example import collate_all

if __name__ == '__main__':

    args.stage2_repeat_times = 1
    args.control_joint = 21
    args.density = 100
    ### stage 1
    args.resume_root = 'output/0518_omni67_multi_partxyz/net_last.pth'; args.roottype = 'omni67'; outname = 'omni67' ; args.normalize_traj=True 
    ### stage 2
    # args.resume_trans = 'output/0509_diffmae_stage2_2_E8D0_multicontrol_pretrain/net_last.pth'; args.modeltype = 'diffmae_stage2_2' 
    args.resume_trans = 'output/0520_semboost/net_last.pth'; args.modeltype = 'semboost'

    clip_model = get_clip_model()

    # 根节点网络
    if args.roottype == 'omni67':
        from models.omni67 import CMDM
        net_root = CMDM(args, args.roottype)
    load_ckpt(net_root, args.resume_root, key='trans')
    net_root.eval()
    net_root.cuda()
    diffusion_root = create_gaussian_diffusion_simple(args, net_root, args.roottype, clip_model)

    # 2阶段网络
    if args.modeltype == 'diffmae_stage2_2':
        from models.diffmae_2 import DiffMAE2
        net = DiffMAE2(dataset=args.dataset_name, args=args, num_layers_E=8, num_layers_D=0)
    elif args.modeltype == 'semboost':
        from models.semanticboost import SemanticBoost
        from utils.model_util import get_semanticboost_args
        net = SemanticBoost(**get_semanticboost_args(args))
    load_ckpt(net, args.resume_trans, key='trans')
    net.eval()
    net.cuda()
    diffusion = create_gaussian_diffusion_simple(args, net, args.modeltype, clip_model)

    #从dataset中随机抽取，取其text与控制joint的263 dim数据
    args.batch_size = 1
    # train_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode='eval', shuffle=False,)
    # train_loader_iter = dataset_control.cycle(train_loader)
    # val_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode='eval', split='val', shuffle=True, num_workers=0)
    # val_loader_iter = dataset_control.cycle(val_loader)
    test_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode='eval', split='test', shuffle=False, num_workers=0, drop_last=True)


    
    for i, batch in enumerate(test_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, gt_motion, real_length, txt_tokens, traj, traj_mask_263, traj_mask = batch
        b, max_length, num_features = gt_motion.shape
        gt_motion = gt_motion.cuda()
        real_length = real_length.cuda()
        traj = traj.cuda()
        traj_mask = traj_mask.cuda()
        traj_mask_263 = traj_mask_263.cuda()
        real_mask = generate_src_mask(max_length, real_length) # (b,196)
        gt_ric = gt_motion[..., :67]

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

        sample, loss_xyz = sample_ADControl(diffusion_root, diffusion,  args, condition, vis=True)
        print(f'loss_xyz = {loss_xyz.item():.4f}')
        break

        # # # 采样根节点轨迹
        # print('=== root sample')
        # if args.roottype == 'omni67': 
        #     pred_ric = diffusion_root.p_sample_loop(partial_emb=None, model_kwargs=condition, batch_size=args.batch_size)
        #     args.use_stage1 = True

        # control_id = traj_mask[0].sum(0).sum(-1).nonzero()
        # print('control_id = ', control_id)
        
        # if args.normalize_traj:
        #     traj = traj * raw_std + raw_mean
        # loss_xyz = calc_loss_xyz(pred_ric, traj, traj_mask) # 仅约束控制的关节误差
        # print(f'loss_xyz = {loss_xyz.item():.4f}')
        
        # # text, traj = collate_all()
        
        # for j in range(args.stage2_repeat_times):
        #     # 采样动作
        #     print(f'=== motion sample: repeat time {j}')
        #     partial_emb = torch.zeros_like(gt_motion, device=gt_motion.device)
        #     partial_emb[..., :67] = gt_ric # 使用dataloader出来的
        #     if args.use_stage1:
        #         partial_emb[..., :67] = pred_ric # 使用1阶段网络预测出来的

        #     if 'diffmae_stage2' in args.modeltype or 'semboost' in args.modeltype:
        #         pred_motion = diffusion.p_sample_loop(partial_emb, with_control=True, model_kwargs=condition, batch_size=args.batch_size) # 有替换
        #         # pred_motion = diffusion.p_sample_loop(partial_emb, with_control=False, model_kwargs=condition,) # 无替换，即纯噪声生成和MDM一样

        #     loss_xyz = calc_loss_xyz(pred_motion, traj, traj_mask)
        #     print(f'loss_xyz = {loss_xyz.item():.4f}')


        #     save_name = f'./output/testsample/{outname}_{j+1}.html'
        #     print('save_name = ', save_name)
        #     visualize_2motions(pred_motion[0].detach().cpu().numpy(), diffusion.std, diffusion.mean, args.dataset_name, None, motion2=gt_motion[0].detach().cpu().numpy(), save_path=save_name)
        #     a = 1

        

    

    
    
    
    
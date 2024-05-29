import os 
import sys
import torch
import numpy as np

from os.path import join as pjoin
import utils.eval_trans as eval_trans
from dataset import dataset_tokenize
from options.get_eval_option import get_opt
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from exit.utils import get_model, visualize_2motions, generate_src_mask, init_save_folder, uniform, cosine_schedule, gumbel_sample
from utils.motion_process import recover_from_ric, recover_root_rot_pos
from einops import rearrange, repeat
import torch.nn.functional as F
import shutil
import utils.losses as losses 
import clip
from utils.mask_utils import random_mask_token, root_dist_loss

def get_acc(cls_pred, target, mask):
    cls_pred = torch.masked_select(cls_pred, mask.unsqueeze(-1)).view(-1, cls_pred.shape[-1])
    target_all = torch.masked_select(target, mask)
    probs = torch.softmax(cls_pred, dim=-1)
    _, cls_pred_index = torch.max(probs, dim=-1)
    right_num = (cls_pred_index == target_all).sum()
    return right_num*100/mask.sum()

def MMMtrans67_trainer_func(args, net_vq, net, clip_model, dataloader_iter, logger, optimizer, scheduler, mean, std):
    for nb_iter in tqdm(range(1,args.total_iter+1), position=0, leave=True):
        batch = next(dataloader_iter)
        clip_text, gt_motion, motion_token, motion_token_len, traj, traj_mask = batch 
        gt_motion = gt_motion.cuda()
        gt_ric = gt_motion[..., :67]
        motion_token = motion_token.cuda()
        motion_token_len = motion_token_len.cuda()
        traj = traj.cuda()
        traj_mask = traj_mask.cuda()

        # CLIP文本
        text = clip.tokenize(clip_text, truncate=True).cuda() 
        text_emb, word_emb = clip_model(text)

        masked_input_indices, real_mask, real_mask_no_end, token_mask = random_mask_token(motion_token, motion_token_len, args)
        # 前向
        cls_pred = net(masked_input_indices,text_emb, traj=traj, src_mask=real_mask, word_emb=word_emb)[1:, ...].permute(1,0,2)
        # cls_pred = net(masked_input_indices,text_emb, traj=traj, src_mask=real_mask, word_emb=word_emb).permute(1,0,2)
        # if args.attn_mask:
        #     cls_pred = net(masked_input_indices, text_emb, traj=traj, src_mask=real_mask, word_emb=word_emb) # (b,50,512)
        # else:
        #     cls_pred = net(masked_input_indices, text_emb, traj=traj, src_mask=None, word_emb=word_emb) # (b,50,512)

        # 计算loss
        loss = 0
        weights = real_mask_no_end / (real_mask_no_end.sum(-1).unsqueeze(-1) * real_mask_no_end.shape[0])
        cls_pred_seq_masked = cls_pred[real_mask_no_end, :].view(-1, cls_pred.shape[-1])
        target_seq_masked = motion_token[real_mask_no_end]
        weight_seq_masked = weights[real_mask_no_end]
        loss_cls = F.cross_entropy(cls_pred_seq_masked, target_seq_masked, reduction = 'none')
        loss_cls = (loss_cls * weight_seq_masked).sum()
        loss += loss_cls

        # 解码回(b,196,67) 然后recover_from_ric
        # pred_idx = gumbel_sample(cls_pred[:, :-1, :], 0, -1) # 去除最后的end token特征
        pred_idx =  cls_pred[:, :-1, :].argmax(-1) # 只取训练中最大概率那个作为预测的idx
        pred_ric = net_vq(pred_idx, type='decode')
        pred_xyz = recover_from_ric(pred_ric * std[..., :67] + mean[..., :67], joints_num=22)
        # 控制关节误差
        loss_xyz_part = F.l1_loss(pred_xyz[traj_mask], traj[traj_mask]) # 仅约束控制轨迹
        if args.loss_xyz: 
            loss += args.loss_xyz * loss_xyz_part 
        # 根误差
        real_motion_mask = generate_src_mask(196, motion_token_len*4)
        loss_rotate_global, loss_position_global, gt_root_pos, pred_root_pos = root_dist_loss(
            gt_ric * std[..., :67] + mean[..., :67], pred_ric * std[..., :67] + mean[..., :67], real_motion_mask)
        if args.root_dist_loss:
            loss += loss_rotate_global
            loss += loss_position_global
        # global loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if nb_iter % args.print_iter ==  0 :
            acc = get_acc(cls_pred,motion_token,real_mask_no_end)

            msg = f'Train. Iter {nb_iter} '
            msg += f' loss_cls. {loss_cls:.4f}'
            msg += f' acc: {acc:.2f}%' 
            msg += f' loss_xyz_part. {loss_xyz_part:.4f} '
            msg += f' loss_global_root. {loss_position_global:.4f} '
            logger.info(msg)
        
        # if nb_iter % args.eval_iter == 0:
        #     loss_rotate, loss_position = calc_root_loss(net_vq, cls_pred, motion_token, motion_token_len, real_mask_motion, mean[..., :4], std[..., :4]) 
        #     msg += f' loss_position. {loss_position:.5f} '
        #     index_motion = net(clip_feature = text_emb,traj = traj, word_emb = word_emb, type="sample", m_length=motion_token_len*4, if_test=False)
        #     acc = (index_motion[0,...,:motion_token_len[0]] == motion_token[0,...,:motion_token_len[0]]).sum() / motion_token_len[0]
        #     msg += f' accsample. {acc:.5f} '
        #     logger.info(msg)

        if nb_iter % args.save_iter == 0:
            torch.save({'trans' : net.state_dict()}, os.path.join(args.out_dir, f'net_last.pth'))
        
        if nb_iter in args.lr_scheduler:
            save_name = os.path.join(args.out_dir, f'iter_{nb_iter}.pth')
            torch.save({'trans' : net.state_dict()}, save_name)

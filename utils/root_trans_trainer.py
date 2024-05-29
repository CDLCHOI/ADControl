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
from utils.mask_utils import random_mask_token

def root_dist_loss(gt_root, pred_root, real_mask):
    gt_rot, gt_pos = recover_root_rot_pos(gt_root)
    pred_rot, pred_pos = recover_root_rot_pos(pred_root)
    loss_rotate = F.l1_loss(gt_rot[real_mask], pred_rot[real_mask], reduction='mean')
    loss_position = F.l1_loss(gt_pos[real_mask], pred_pos[real_mask], reduction='mean')
    return loss_rotate, loss_position, gt_pos, pred_pos

def calc_save_motion_token(args, net_vq):
    codebook_dir = f'{args.vq_dir}/codebook/'
    os.makedirs(codebook_dir, exist_ok = True)
    if len(os.listdir(codebook_dir)) == 0:
    # if True:
        # 默认2次下采样。unit_length=4   dataloader里面把原本checkpoint的meanstd改为了数据集里的mean std
        train_loader_token = dataset_tokenize.DATALoader(args.dataname, 1, unit_length=2**args.down_t) 
        print('encode motion to token idx')
        for batch in tqdm(train_loader_token):
            pose, name = batch
            bs, seq = pose.shape[0], pose.shape[1]
            pose = pose.cuda().float() # bs, nb_joints, joints_dim, seq_len
            pose = pose[..., :67] if args.modeltype == 'MMMtrans67' else pose[..., :4]
            token = net_vq(pose, type='encode')
            # pred = net_vq(token, type='decode')
            # print('loss = ', F.l1_loss(pose, pred))
            token = token.cpu().numpy()
            if not os.path.exists(pjoin(codebook_dir, name[0] +'.npy')):
                np.save(pjoin(codebook_dir, name[0] +'.npy'), token)
    else:
        print('=== token already exists')

def get_acc(cls_pred, target, mask):
    cls_pred = torch.masked_select(cls_pred, mask.unsqueeze(-1)).view(-1, cls_pred.shape[-1])
    target_all = torch.masked_select(target, mask)
    probs = torch.softmax(cls_pred, dim=-1)
    _, cls_pred_index = torch.max(probs, dim=-1)
    right_num = (cls_pred_index == target_all).sum()
    return right_num*100/mask.sum()

def root_trans_trainer_func(args, net_vq, net, clip_model, dataloader_iter, logger, optimizer, scheduler, mean, std):
    avg_loss_cls = 0
    for nb_iter in tqdm(range(1,args.total_iter+1), position=0, leave=True):
        batch = next(dataloader_iter)
        clip_text, gt_motion, motion_token, motion_token_len, traj, traj_mask = batch 
        motion_token = motion_token.cuda()
        motion_token_len = motion_token_len.cuda()
        traj = traj.cuda()
        traj_mask = traj_mask.cuda()

        # pred_motion = net.vqvae.forward(motion_token, type='decode')
        # visualize_2motions(pred_motion[0].detach().cpu().numpy(), mean, 
        #                std, 't2m', None, motion2=None, save_path=pjoin(args.out_dir, 'text.html'))
        # break

        # CLIP文本
        text = clip.tokenize(clip_text, truncate=True).cuda() 
        text_emb, word_emb = clip_model(text)

        masked_input_indices, real_mask, real_mask_no_end, mask_token = random_mask_token(motion_token, motion_token_len, args)
        # 前向
        if args.modeltype == 'root_trans':
            cls_pred = net(masked_input_indices, text_emb, traj=traj, src_mask=real_mask, word_emb=word_emb)[1:, ...].permute(1,0,2)
        elif args.modeltype == 'root_trans2':
            if args.attn_mask:
                cls_pred = net(masked_input_indices, text_emb, traj=traj, src_mask=real_mask, word_emb=word_emb) # (b,50,512)
            else:
                cls_pred = net(masked_input_indices, text_emb, traj=traj, src_mask=None, word_emb=word_emb) # (b,50,512)

        # 计算loss
        weights = real_mask_no_end / (real_mask_no_end.sum(-1).unsqueeze(-1) * real_mask_no_end.shape[0])
        cls_pred_seq_masked = cls_pred[real_mask_no_end, :].view(-1, cls_pred.shape[-1])
        target_seq_masked = motion_token[real_mask_no_end]
        weight_seq_masked = weights[real_mask_no_end]
        loss_cls = F.cross_entropy(cls_pred_seq_masked, target_seq_masked, reduction = 'none')
        loss_cls = (loss_cls * weight_seq_masked).sum()

        real_mask_motion = generate_src_mask(196, motion_token_len*4)

        ## global loss
        optimizer.zero_grad()
        loss_cls.backward()
        optimizer.step()
        scheduler.step()

        avg_loss_cls += loss_cls.item()

        if nb_iter % args.print_iter ==  0 :
            avg_loss_cls /= args.print_iter
            acc = get_acc(cls_pred, motion_token, mask_token)
            msg = f'Train. Iter {nb_iter} '
            msg += f' loss_cls. {loss_cls:.5f}  acc. {acc:.5f}'
            logger.info(msg)
            avg_loss_cls = 0
        if nb_iter % args.eval_iter == 0:
            loss_rotate, loss_position = calc_root_loss(net_vq, cls_pred, motion_token, motion_token_len, real_mask_motion, mean[..., :4], std[..., :4]) 
            msg += f' loss_position. {loss_position:.5f} '
            logger.info(msg)


        if nb_iter % args.save_iter == 0:
            torch.save({'trans' : net.state_dict()}, os.path.join(args.out_dir, 'net_last.pth'))
        
        if nb_iter in args.lr_scheduler:
            save_name = os.path.join(args.out_dir, f'iter_{nb_iter}.pth')
            torch.save({'trans' : net.state_dict()}, save_name)

def calc_root_loss(net_vq, pred_vec, gt_idx, motion_token_len, real_mask_motion, mean, std ):
    '''
    pred_vec: (b,50,512)
    gt_idx: (b,50)
    real_mask_no_end: (b,50)
    '''
    pred_idx = gumbel_sample(pred_vec[:, :-1, :], 0, -1) # 去除最后的end token特征
    pred_root = net_vq(pred_idx, type='decode')
    gt_root = torch.zeros_like(pred_root, device=pred_root.device).float()
    B, L = gt_idx.shape
    for b in range(B):
        tmp = net_vq(gt_idx[b:b+1, :motion_token_len[b]], type='decode')
        gt_root[b:b+1, :motion_token_len[b]*4] = tmp

    loss_rotate, loss_position, gt_pos, pred_pos = root_dist_loss(pred_root * std + mean, gt_root * std + mean, real_mask_motion)
    return loss_rotate, loss_position

def eval_transformer():
    pass

def sample(self, text_emb, word_emb, motion_length=None, if_test=False, rand_pos=True, CFG=-1, token_cond=None, max_steps = 10):
    max_length = 49
    batch_size = text_emb.shape[0]
    mask_id = self.num_vq + 2 # 8192 + 2
    pad_id = self.num_vq + 1 # 8192 + 1
    end_id = self.num_vq # 8192
    shape = (batch_size, self.block_size - 1) # (1, 50)
    topk_filter_thres = .9
    starting_temperature = 1.0
    scores = torch.ones(shape, dtype = torch.float32, device = text_emb.device)
    
    motion_token_len = torch.ceil((m_length)/4).long() # T/4
    src_token_mask = generate_src_mask(self.block_size-1, m_tokens_len+1) # 网络支持最大token长度50，生成用户指定长度/4的mask
    src_token_mask_noend = generate_src_mask(self.block_size-1, m_tokens_len) # 
    if token_cond is not None:
        ids = token_cond.clone()
        ids[~src_token_mask_noend] = pad_id
        num_token_cond = (ids==mask_id).sum(-1)
    else:
        ids = torch.full(shape, mask_id, dtype = torch.long, device = text_emb.device)
    
    # [TODO] confirm that these 2 lines are not neccessary (repeated below and maybe don't need them at all)
    ids[~src_token_mask] = pad_id # [INFO] replace with pad id
    ids.scatter_(-1, m_tokens_len[..., None].long(), end_id) # [INFO] replace with end id

    sample_max_steps = torch.round(max_steps/max_length*m_tokens_len) + 1e-8
    for step in range(max_steps):
        timestep = torch.clip(step/(sample_max_steps), max=1)
        if len(m_tokens_len)==1 and step > 0 and torch.clip(step-1/(sample_max_steps), max=1).cpu().item() == timestep:
            break
        rand_mask_prob = cosine_schedule(timestep) # timestep #
        num_token_masked = (rand_mask_prob * m_tokens_len).long().clip(min=1)

        if token_cond is not None:
            num_token_masked = (rand_mask_prob * num_token_cond).long().clip(min=1)
            scores[token_cond!=mask_id] = 0
        
        # [INFO] rm no motion frames
        scores[~src_token_mask_noend] = 0
        scores = scores/scores.sum(-1)[:, None] # normalize only unmasked token
        
        if rand_pos:
            sorted_score_indices = scores.multinomial(scores.shape[-1], replacement=False) # stocastic
        else:
            sorted, sorted_score_indices = scores.sort(descending=True) # 返回降序的values, indices  (b,50)
        
        ids[~src_token_mask] = pad_id # [INFO] replace with pad id
        ids.scatter_(-1, m_tokens_len[..., None].long(), end_id) # [INFO] replace with end id 把end位置的id换成end_id
        ## [INFO] Replace "mask_id" to "ids" that have highest "num_token_masked" "scores" 
        select_masked_indices = generate_src_mask(sorted_score_indices.shape[1], num_token_masked)
        # [INFO] repeat last_id to make it scatter_ the existing last ids.
        last_index = sorted_score_indices.gather(-1, num_token_masked.unsqueeze(-1)-1)
        sorted_score_indices = sorted_score_indices * select_masked_indices + (last_index*~select_masked_indices)
        ids.scatter_(-1, sorted_score_indices, mask_id) # 把ids中 sorted_score_indices 位置的值替换为mask_id。即网络预测后的remask操作，作者在实现上放到这里了

        logits = self.forward(ids, text_emb, src_token_mask, word_emb=word_emb)[:,1:]
        filtered_logits = logits #top_p(logits, .5) # #top_k(logits, topk_filter_thres)
        if rand_pos:
            temperature = 1 #starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed
        else:
            temperature = 0 #starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed

        # [INFO] if temperature==0: is equal to argmax (filtered_logits.argmax(dim = -1))
        # pred_ids = filtered_logits.argmax(dim = -1)
        pred_ids = gumbel_sample(filtered_logits, temperature = temperature, dim = -1) # (1,50)
        is_mask = ids == mask_id

        ids = torch.where(
                    is_mask,
                    pred_ids,
                    ids
                )
        
        # if timestep == 1.:
        #     print(probs_without_temperature.shape)
        probs_without_temperature = logits.softmax(dim = -1)
        scores = 1 - probs_without_temperature.gather(-1, pred_ids[..., None])
        scores = rearrange(scores, '... 1 -> ...')
        scores = scores.masked_fill(~is_mask, 0)
    if if_test:
        return ids
    return ids

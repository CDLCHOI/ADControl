
import numpy as np
import torch
import torch.nn.functional as F
from os.path import join as pjoin
from tqdm import tqdm
import clip

from exit.utils import generate_src_mask
from utils.mask_utils import root_dist_loss, TextCLIP, recover, vis_motion, calc_loss_xyz
from utils.motion_process import recover_from_ric, recover_root_rot_pos
from utils.metrics import evaluate_control, evaluate_control_diffmae, evaluate_control_whentraining
from diffusion.resample import create_named_schedule_sampler

class GaussianDiffusionSimple:
    def __init__(self, args, model, modeltype, clip_model, betas) -> None:
        self.args = args
        self.model = model
        self.modeltype = modeltype # 'ED'
        self.clip_model = clip_model

        if self.args.dataname == 't2m':
            self.n_joints = 22
            self.mean = torch.from_numpy(np.load('dataset/HumanML3D/Mean.npy')).cuda()[None, None, ...] # dataset/HumanML3D/Mean.npy
            self.std = torch.from_numpy(np.load('dataset/HumanML3D/Std.npy')).cuda()[None, None, ...]
            self.raw_mean = torch.from_numpy(np.load('dataset/humanml_spatial_norm/Mean_raw.npy')).cuda()[None, None, ...].view(1,1,22,3) 
            self.raw_std = torch.from_numpy(np.load('dataset/humanml_spatial_norm/Std_raw.npy')).cuda()[None, None, ...].view(1,1,22,3)
        elif self.args.dataname == 'kit':
            self.n_joints = 21
            self.mean = torch.from_numpy(np.load('dataset/KIT/Mean.npy')).cuda()[None, None, ...] # dataset/HumanML3D/Mean.npy
            self.std = torch.from_numpy(np.load('dataset/KIT/Std.npy')).cuda()[None, None, ...]

        # diffusion相关参数值
        betas = np.array(betas, dtype=np.float64) # 每个step的噪声方差，如果总共有T个step，那betas长度就是T
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        # 这一部分用于前向加噪过程
        alphas = 1.0 - betas # (1000,)
        self.alphas_cumprod = np.cumprod(alphas, axis=0) # alpha t的累乘 # (1000,)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1]) # alpha t-1的累乘 # (1000,)
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # 这一部分用于反向去噪过程
        # calculations for diffusion q(xt | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod) # DDPM原文公式（4）的x0系数  根号(alpha的累乘)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod) # 公式（3）的噪声系数
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | xt, x_0)
        
        self.posterior_variance = ( 
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod) # 对应公式（7）中beta_t波浪
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = ( 
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod) # 公式（7）中后验均值 x_0的系数
        )
        self.posterior_mean_coef2 = ( # 公式（7）中后验均值 x_t的系数
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
        
        self.schedule_sampler = create_named_schedule_sampler('uniform', self)

    
    
    def trainer_func(self, dataloader_iter, logger, optimizer, scheduler):
        for nb_iter in tqdm(range(1, self.args.total_iter+1), position=0, leave=True):
            batch = next(dataloader_iter)
            word_embeddings, pos_one_hots, clip_text, sent_len, gt_motion, real_length, txt_tokens, traj, traj_mask_263, traj_mask = batch
            # clip_text, gt_token, m_tokens_len = batch
            b, max_length, num_features = gt_motion.shape
            gt_motion = gt_motion.cuda()
            real_length = real_length.cuda()
            traj = traj.cuda()
            traj_mask_263 = traj_mask_263.cuda()
            traj_mask = traj_mask.cuda()
            real_mask = generate_src_mask(max_length, real_length) # (b,196)
            
            text = clip.tokenize(clip_text, truncate=True).cuda()        
            text_emb, word_emb = self.clip_model(text) # (b,512)  (b,77,512)

            t, weights = self.schedule_sampler.sample(b, gt_motion.device) # timestep
            # t = torch.tensor([900]*b).cuda()
            x0 = gt_motion
            noise = torch.randn_like(x0) # 生成与x0形状一样的高斯噪声
            xt = self.q_sample(x0, t, noise=noise) # 给数据集x0加t步噪声
            # xt = self.guide(xt, t, model_kwargs=model_kwargs, train=True)
            if self.args.no_random_mask:
                masked_xt = xt
            else:
                if np.random.choice([0,1]):
                    masked_xt = torch.where(traj_mask_263, gt_motion, xt) # mask以外的部分恢复原数值
                else:
                    masked_xt = xt 
            # 前向
            if 'diffmae_stage2' in self.modeltype:
                pred_x0 = self.model(masked_xt, t, text_emb, word_emb) # (b,196,263)
            elif self.modeltype == 'diffmdm':
                xt = xt.permute(0,2,1)[:,:,None]
                pred_x0 = self.model(xt, t, y={'text':clip_text, 'hint': traj.flatten(2,3)})  # (b,196,263)
                pred_x0 = pred_x0.squeeze(2).permute(0,2,1)

            loss, msg = self.calc_loss(x0, pred_x0, self.mean, self.std, traj_mask, traj, real_mask, traj_mask_263, num_features, nb_iter)
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if nb_iter % self.args.print_iter ==  0 :
                logger.info(msg)

            if nb_iter % self.args.save_iter == 0:
                torch.save({'trans' : self.model.state_dict()}, pjoin(self.args.out_dir, 'net_last.pth'))
            
    def trainer_func_root67(self, dataloader_iter, logger, optimizer, scheduler):

        for nb_iter in tqdm(range(1, self.args.total_iter+1), position=0, leave=True):
            batch = next(dataloader_iter)
            word_embeddings, pos_one_hots, clip_text, sent_len, gt_motion, real_length, txt_tokens, traj, traj_mask_263, traj_mask = batch
            # clip_text, gt_token, m_tokens_len = batch
            gt_motion = gt_motion.cuda()
            gt_ric = gt_motion[..., :67]
            b, max_length, num_features = gt_ric.shape
            real_length = real_length.cuda()
            traj = traj.cuda()
            traj_mask_263 = traj_mask_263.cuda()
            traj_mask = traj_mask.cuda()
            real_mask = generate_src_mask(max_length, real_length) # (b,196)
            ric_mask_67 = traj_mask_263[..., :67]
            

            text = clip.tokenize(clip_text, truncate=True).cuda()        
            text_emb, word_emb = self.clip_model(text) # (b,512)  (b,77,512)

            condition = {}
            condition['traj'] = traj
            condition['text_emb'] = text_emb
            condition['word_emb'] = word_emb
            condition['traj_mask'] = traj_mask
            condition['traj_mask_263'] = traj_mask_263
            condition['gt_motion'] = gt_motion
            condition['traj'] = traj
            condition['real_mask'] = real_mask

            t, weights = self.schedule_sampler.sample(b, gt_ric.device) # timestep
            # t = torch.tensor([900]*b).cuda()
            x0 = gt_ric
            noise = torch.randn_like(x0) # 生成与x0形状一样的高斯噪声
            xt = self.q_sample(x0, t, noise=noise) # 给数据集x0加t步噪声
            xt = self.guide(xt, t, condition, train=True)
            # 前向
            pred_x0 = self.model(xt, t, text_emb, traj=traj)  # (b,196,263)
            loss, msg = self.calc_loss(x0, pred_x0, self.mean[..., :67], self.std[..., :67], traj_mask, traj, real_mask, traj_mask_263, num_features, nb_iter)
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if nb_iter % self.args.print_iter ==  0 :
                logger.info(msg)

            if nb_iter % self.args.save_iter == 0:
                torch.save({'trans' : self.model.state_dict()}, pjoin(self.args.out_dir, 'net_last.pth'))

    def trainer_func_omni67(self, dataloader_iter, logger, optimizer, scheduler, test_loader=None):
        min_err = 100
        for nb_iter in tqdm(range(1, self.args.total_iter+1), position=0, leave=True):
            batch = next(dataloader_iter)
            word_embeddings, pos_one_hots, clip_text, sent_len, gt_motion, real_length, txt_tokens, traj, traj_mask_263, traj_mask = batch
            # clip_text, gt_token, m_tokens_len = batch
            gt_motion = gt_motion.cuda()
            gt_ric = gt_motion[..., :67]
            b, max_length, num_features = gt_ric.shape
            real_length = real_length.cuda()
            traj = traj.cuda()
            traj_mask_263 = traj_mask_263.cuda()
            traj_mask = traj_mask.cuda()
            real_mask = generate_src_mask(max_length, real_length) # (b,196)
            ric_mask_67 = traj_mask_263[..., :67]
            

            text = clip.tokenize(clip_text, truncate=True).cuda()        
            text_emb, word_emb = self.clip_model(text) # (b,512)  (b,77,512)

            condition = {}
            condition['clip_text'] = clip_text
            condition['traj'] = traj
            condition['text_emb'] = text_emb
            condition['word_emb'] = word_emb
            condition['traj_mask'] = traj_mask
            condition['traj_mask_263'] = traj_mask_263
            condition['gt_motion'] = gt_motion
            condition['traj'] = traj
            condition['real_mask'] = real_mask

            t, weights = self.schedule_sampler.sample(b, gt_ric.device) # timestep
            # t = torch.tensor([900]*b).cuda()
            x0 = gt_ric
            noise = torch.randn_like(x0) # 生成与x0形状一样的高斯噪声
            xt = self.q_sample(x0, t, noise=noise) # 给数据集x0加t步噪声
            xt = self.guide(xt, t, condition, train=True)
            # 前向
            xt = xt.permute(0,2,1)[:,:,None]
            pred_x0 = self.model(xt, t, y={'text':clip_text, 'hint': traj.flatten(2,3)})  # (b,196,263)
            pred_x0 = pred_x0.squeeze(2).permute(0,2,1)
            loss, msg = self.calc_loss(x0, pred_x0, self.mean[..., :67], self.std[..., :67], traj_mask, traj, real_mask, traj_mask_263, num_features, nb_iter)
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if nb_iter % self.args.print_iter ==  0 :
                logger.info(msg)

            if nb_iter % self.args.save_iter == 0:
                err, _, _ = evaluate_control_whentraining(test_loader, self.clip_model, self, self.mean, self.std, self.args, logger, batch_size=128)
                logger.info(f'err = {err:.4f}, min_err = {min_err:.4f}')
                torch.save({'trans' : self.model.state_dict()}, pjoin(self.args.out_dir, 'net_last.pth'))
                if err < min_err:
                    min_err = err
                    torch.save({'trans' : self.model.state_dict()}, pjoin(self.args.out_dir, 'net_best.pth'))

    def trainer_func_semboost(self, dataloader_iter, logger, optimizer, scheduler):
        for nb_iter in tqdm(range(1, self.args.total_iter+1), position=0, leave=True):
            batch = next(dataloader_iter)
            word_embeddings, pos_one_hots, clip_text, sent_len, gt_motion, real_length, txt_tokens, traj, traj_mask_263, traj_mask = batch
            # clip_text, gt_token, m_tokens_len = batch
            b, max_length, num_features = gt_motion.shape
            gt_motion = gt_motion.cuda()
            real_length = real_length.cuda()
            traj = traj.cuda()
            traj_mask_263 = traj_mask_263.cuda()
            traj_mask = traj_mask.cuda()
            real_mask = generate_src_mask(max_length, real_length) # (b,196)
        
            # text = clip.tokenize(clip_text, truncate=True).cuda()        
            # text_emb, word_emb = self.clip_model(text) # (b,512)  (b,77,512)

            t, weights = self.schedule_sampler.sample(b, gt_motion.device) # timestep
            # t = torch.tensor([999]*b).cuda() # debug用
            x0 = gt_motion
            noise = torch.randn_like(x0) # 生成与x0形状一样的高斯噪声
            xt = self.q_sample(x0, t, noise=noise) # 给数据集x0加t步噪声
            # xt = self.guide(xt, t, model_kwargs=model_kwargs, train=True)
            if self.args.no_random_mask:
                masked_xt = xt
            else:
                if np.random.choice([0,1]):
                    masked_xt = torch.where(traj_mask_263, gt_motion, xt) # mask以外的部分恢复原数值
                else:
                    masked_xt = xt 
            # 前向
            masked_xt = masked_xt.permute(0,2,1)[:,:,None]
            # masked_xt= torch.ones((1,263,1,120)).cuda()*0.5 # debug用
            y={'text': clip_text}
            pred_x0 = self.model(masked_xt, t, y=y)  # (b,196,263)
            pred_x0 = pred_x0.squeeze(2).permute(0,2,1)
            loss, msg = self.calc_loss(x0, pred_x0, self.mean, self.std, traj_mask, traj, real_mask, traj_mask_263, num_features, nb_iter)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if nb_iter % self.args.print_iter ==  0 :
                logger.info(msg)

            if nb_iter % self.args.save_iter == 0:
                torch.save({'trans' : self.model.state_dict()}, pjoin(self.args.out_dir, 'net_last.pth'))

    def calc_loss(self, gt, pred, mean, std, traj_mask, traj, real_mask, traj_mask_263, num_features, nb_iter):
        loss = 0
        if self.args.normalize_traj:
            traj = traj * self.raw_std + self.raw_mean
        recon_xyz = recover_from_ric(pred * std + mean, joints_num=22)  # 反归一化再转全局xyz
        gt_xyz = recover_from_ric(gt * std + mean, joints_num=22)
        assert torch.allclose(gt_xyz * traj_mask, traj * traj_mask, atol=1e-5) # 确保轨迹及mask是正确的

        # 损失函数计算
        motion_real_mask = real_mask[..., None].repeat(1,1, num_features)
        xyz_real_mask = real_mask[..., None, None].repeat(1,1,22,3)
        if self.args.loss_type == 'l1':
            loss_motion = F.l1_loss(pred[motion_real_mask], gt[motion_real_mask])
            loss_xyz_all = F.l1_loss(recon_xyz[xyz_real_mask], gt_xyz[xyz_real_mask]) # 约束全身轨迹  都替换为L1Loss
            loss_xyz_part = F.l1_loss(recon_xyz[traj_mask], traj[traj_mask]) # 仅约束控制轨迹
        elif self.args.loss_type == 'l2':
            loss_motion = F.mse_loss(pred[motion_real_mask], gt[motion_real_mask])
            loss_xyz_all = F.mse_loss(recon_xyz[xyz_real_mask], gt_xyz[xyz_real_mask]) # 约束全身轨迹  都替换为L1Loss
            loss_xyz_part = F.mse_loss(recon_xyz[traj_mask], traj[traj_mask]) # 仅约束控制轨迹

        gt_root = (gt * std + mean)[..., :4]
        recon_root = (pred * std + mean)[..., :4]
        loss_rotate_global, loss_position_global, gt_root_pos, pred_root_pos = root_dist_loss(gt_root, recon_root, real_mask, self.args)

        if self.args.root_dist_loss:
            loss += loss_rotate_global
            loss += loss_position_global
        # assert torch.allclose(gt_root_pos, gt_xyz[:,:,0,:])


        # if text2motion_flag:
        #     self.args.xyz_type = 'all' # text to motion情况使用全部关节点约束
        if self.args.xyz_type == 'all':
            loss_xyz = loss_xyz_all
        if self.args.xyz_type == 'part':
            loss_xyz = loss_xyz_part
            
        if self.args.loss_xyz: 
            loss += self.args.loss_xyz * loss_xyz
        loss = loss + loss_motion

        msg = f'Train. Iter {nb_iter} '
        msg += f" loss_motion. {loss_motion:.4f}, loss_xyz. {loss_xyz:.4f} "
        msg += f' loss_rotate_global. {loss_rotate_global:.4f}, loss_position_global. {loss_position_global:.4f} '
        return loss, msg

    def p_sample_single(self, dataloader_iter):
        # mean and std
        if self.args.dataname == 't2m':
            mean = torch.from_numpy(np.load('dataset/HumanML3D/Mean.npy')).cuda()[None, None, ...] # dataset/HumanML3D/Mean.npy
            std = torch.from_numpy(np.load('dataset/HumanML3D/Std.npy')).cuda()[None, None, ...]
        elif self.args.dataname == 'kit':
            mean = torch.from_numpy(np.load('dataset/HumanML3D/Mean.npy')).cuda()[None, None, ...] # dataset/HumanML3D/Mean.npy
            std = torch.from_numpy(np.load('dataset/HumanML3D/Std.npy')).cuda()[None, None, ...]

        for nb_iter in tqdm(range(1, self.args.total_iter+1), position=0, leave=True):
            batch = next(dataloader_iter)
            word_embeddings, pos_one_hots, clip_text, sent_len, gt_motion, real_length, txt_tokens, traj, traj_mask_263, traj_mask = batch
            # clip_text, gt_token, m_tokens_len = batch
            b, max_length, num_features = gt_motion.shape
            gt_motion = gt_motion.cuda()
            real_length = real_length.cuda()
            traj = traj.cuda()
            traj_mask_263 = traj_mask_263.cuda()
            traj_mask = traj_mask.cuda()
            real_mask = generate_src_mask(max_length, real_length) # (b,196)
            
            text = clip.tokenize(clip_text, truncate=True).cuda()        
            text_emb, word_emb = self.clip_model(text) # (b,512)  (b,77,512)

            # t, weights = self.schedule_sampler.sample(b, gt_motion.device) # timestep
            t = torch.tensor([999]*b).cuda()
            x0 = gt_motion
            noise = torch.randn_like(x0) # 生成与x0形状一样的高斯噪声
            # xt = self.q_sample(x0, t, noise=noise) # 给数据集x0加t步噪声
            masked_xt = torch.where(traj_mask_263, gt_motion, noise) # mask以外的部分恢复原数值
            # 前向
            pred_x0 = self.model(masked_xt, t, text_emb, word_emb) # (b,196,263)
            loss, msg = self.calc_loss(x0, pred_x0, mean, std, traj_mask, traj, real_mask, traj_mask_263, num_features, nb_iter)
            print(msg)
            a = 1

    def p_sample_loop(self, partial_emb, with_control=True, model_kwargs=None, batch_size=1, control_once=False):
        '''
        partial_emb: (b,196,263)
        condition: 字典，包含文本条件和轨迹条件
        '''
        B = batch_size # batch_size
        skip_t = 0
        indices = list(range(self.num_timesteps - skip_t))[::-1]

        # 初始化噪声方式1
        # noise = torch.randn(*(partial_emb.shape)).cuda()
        # zero = torch.zeros_like(partial_emb).cuda()
        # my_t = torch.ones([B], dtype=torch.long).cuda() * indices[0]
        # xt = self.q_sample(zero, my_t, noise)

        # 初始化噪声方式2
        if 'diffmae_stage2' in self.modeltype:
            noise = torch.randn((B,196,263)).cuda()
        elif self.modeltype in ['diffmae_root67', 'omni67'] :
            noise = torch.randn((B,196,67)).cuda()

        # 对gt_ric加噪得到xt  做测试用
        # gt_ric = model_kwargs['gt_motion'][..., :67]
        # my_t = torch.ones([B], dtype=torch.long).cuda() * indices[0]
        # xt = self.q_sample(gt_ric, my_t)
        # 纯噪声开始
        xt = noise
        if control_once:
            xt = torch.where(model_kwargs['traj_mask_263'], partial_emb, xt) # 消融实验用
        with torch.no_grad():
            for i in tqdm(indices): # 999 ~ 0
                if 'diffmae_stage2' in self.modeltype and with_control:
                    xt = torch.where(model_kwargs['traj_mask_263'], partial_emb, xt) # 2阶段才要做替换
                
                # assert torch.allclose(xt*traj_mask_263, partial_emb)
                t = torch.tensor([i] * B).cuda() # timestep tensor
                out = self.p_sample(xt, t, partial_emb, model_kwargs=model_kwargs) # 返回x_{t-1}和x0
                xt = out["sample"] #上面返回的x_{t-1}
        
        if 'diffmae_stage2' in self.modeltype and with_control:
            out['pred_x0'] = torch.where(model_kwargs['traj_mask_263'], partial_emb, out['pred_x0']) # 2阶段才要做替换
        return out['pred_x0']

    def p_sample(self, xt, t, partial_emb, model_kwargs=None):
        '''
        返回 x_{t-1}
        '''
        B = xt.shape[0]
        out = self.p_mean_variance(xt, t, model_kwargs=model_kwargs) 

        # pred_x0 = out['pred_x0']
        # gt_ric = model_kwargs['gt_motion'][..., :67]
        # num_features = 67
        # motion_real_mask = model_kwargs['real_mask'][..., None].repeat(1, 1, num_features)
        # loss_motion = F.l1_loss(pred_x0[motion_real_mask], gt_ric[motion_real_mask])

        if self.modeltype in ['diffmae_root67', 'omni67'] :
            out['mean'] = self.guide(out['mean'], t, condition=model_kwargs)
        if 'diffmae_stage2' in self.modeltype:
            xt = torch.where(model_kwargs['traj_mask_263'], partial_emb, xt) # 2阶段才要做替换
        mean = out['mean']   
        var = out['variance']
        log_var = out['log_variance']
        pred_x0 = out['pred_x0']

        noise = torch.randn_like(xt)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(xt.shape) - 1))) # no noise when t == 0
        # noise是标准正态分布，sample是通过预测噪声，计算得到方差，再通过重参数化采样得到的x_{t-1}
        sample = mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

        # traj_mask_263 = model_kwargs['traj_mask_263']
        # traj_mask = model_kwargs['traj_mask']
        # traj = model_kwargs['traj']
        num_features = pred_x0.shape[-1]
        motion_real_mask = model_kwargs['real_mask'][..., None].repeat(1, 1, num_features)
        # # xyz_real_mask = real_mask[..., None, None].repeat(1,1,22,3)
        # # recon_xyz = recover_from_ric(pred_x0 * humanml_std + humanml_mean, joints_num=22)  # 反归一化再转全局xyz
        # # gt_xyz = recover_from_ric(gt_motion * humanml_std + humanml_mean, joints_num=22)
        loss_motion = F.l1_loss(pred_x0[motion_real_mask], xt[motion_real_mask])
        # loss_motion2 = F.l1_loss(pred_x0[traj_mask_263], xt[traj_mask_263])
        # # loss_xyz_part = F.l1_loss(recon_xyz[traj_mask], traj[traj_mask]) 
        # if t==0:
        #     a = 1
        #     pred_xyz = recover(pred_x0)
        #     F.l1_loss(xt[traj_mask_263],pred_x0[traj_mask_263])
        #     F.l1_loss(traj[traj_mask],pred_xyz[traj_mask])

        return {"sample": sample, 
                "pred_x0": pred_x0,
                'loss_motion': loss_motion} # 分别是x_{t-1}和x_0


    def p_mean_variance(self, masked_xt, t, model_kwargs=None):
        B = masked_xt.shape[0]
        assert t.shape == (B,)
        # 网络的前向，得到预测的x0
        # model_output = model(x, self._scale_timesteps(t), **model_kwargs) # model_kwargs['y']['mask'].shape=（1，1，1，120）
        text_emb = model_kwargs['text_emb']
        word_emb = model_kwargs['word_emb']
        traj = model_kwargs['traj']
        clip_text = model_kwargs['clip_text']
        # 前向推理
        if 'diffmae_stage2' in self.modeltype:
            pred_x0 = self.model(masked_xt, t, text_emb, word_emb) 
        elif self.modeltype == 'diffmae_root67':
            pred_x0 = self.model(masked_xt, t, text_emb, traj=traj) 
        elif self.modeltype == 'omni67':
            xt = masked_xt.permute(0,2,1)[:,:,None]
            pred_x0 = self.model(xt, t, y={'text':clip_text, 'hint': traj.flatten(2,3)})
            pred_x0 = pred_x0.squeeze(2).permute(0,2,1)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped
        model_variance = _extract_into_tensor(model_variance, t, masked_xt.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, masked_xt.shape)

        # 得到x0后去算x_{t-1}的均值，即后验均值 q(x_{t-1} | x_t, x_0)
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_x0, x_t=masked_xt, t=t) 

        assert model_mean.shape == model_log_variance.shape == pred_x0.shape == masked_xt.shape
        
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_x0": pred_x0,
        }
    

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        assert noise.shape == x0.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
            * noise
        )
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        DDPM原论文公式(7) q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = ( # 公式（7）中的mu_t就是这个后验均值
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def guide(self, x, t, condition, t_stopgrad=-10, scale=.5, n_guide_steps=10, train=False, min_variance=0.01):
        """
        Spatial guidance
        """
        n_joint = 22 if x.shape[-1] == 67 else 21
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        model_variance = torch.exp(model_log_variance)
        
        if model_variance[0, 0, 0] < min_variance:
            model_variance = min_variance

        if train:
            if t[0] < 20:
                n_guide_steps = 100
            else:
                n_guide_steps = 20
        else:
            if t[0] < 10:
                n_guide_steps = 500
            else:
                n_guide_steps = 10

        mask_hint = condition['traj_mask']
        hint = condition['traj'].clone().detach()
        if self.args.normalize_traj:
            # process hint
            if self.raw_std.device != hint.device:
                self.raw_mean = self.raw_mean.to(hint.device)
                self.raw_std = self.raw_std.to(hint.device)
                self.mean = self.mean.to(hint.device)
                self.std = self.std.to(hint.device)
            # 判断是否外部给定了mean和std，在测试集时候使用
            mean = condition.get('mean', None)
            std = condition.get('std', None)
            if mean is None and std is None:
                mean = self.mean
                std = self.std
            hint = hint * self.raw_std + self.raw_mean
            # hint = hint.view(hint.shape[0], hint.shape[1], n_joint, 3) * mask_hint

        
        if not train:
            scale = self.calc_grad_scale(mask_hint[..., :1]) # omnicontrol这里的mask输入shape是 (b,196,22,1)
            # a = torch.linspace(1, 3, steps=196).to(mask_hint.device)
            # weight = (a**1)[None, :, None]
            scale = scale * 3

        for _ in range(n_guide_steps):
            loss, grad = self.gradients(x, self.mean[..., :67], self.std[..., :67], hint, mask_hint) # x和hint都是未归一化的
            grad = model_variance * grad
            if t[0] >= t_stopgrad:
                x = x - scale * grad
        return x.detach()
    
    def calc_grad_scale(self, mask_hint):
        assert mask_hint.shape[1] == 196
        num_keyframes = mask_hint.sum(dim=1).squeeze(-1)
        max_keyframes = num_keyframes.max(dim=1)[0]
        scale = 20 / max_keyframes
        if self.modeltype == 'omni67':
            return scale.unsqueeze(-1).unsqueeze(-1)
        else:
            return scale

    def gradients(self, x, mean, std, hint, mask_hint, joint_ids=None):
        with torch.enable_grad():
            x.requires_grad_(True)
            x_ = x * std + mean
            n_joints = 22 if x_.shape[-1] == 67 else 21
            joint_pos = recover_from_ric(x_, n_joints) # 全局xyz
            if n_joints == 21: # 猜测是KIT格式, 就要把 毫米转为米？
                joint_pos = joint_pos * 0.001
                hint = hint * 0.001

            loss = torch.norm((joint_pos - hint) * mask_hint, dim=-1)
            grad = torch.autograd.grad([loss.sum()], [x])[0] # （b, l, 67）
            # the motion in HumanML3D always starts at the origin (0,y,0), so we zero out the gradients for the root joint
            grad[:, 0, :] = 0 # 第0帧梯度置0
            x.detach()
        return loss, grad


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
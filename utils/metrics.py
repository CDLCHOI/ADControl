# This code is based on https://github.com/GuyTevet/motion-diffusion-model
import numpy as np
from scipy import linalg
from scipy.ndimage import uniform_filter1d
from collections import OrderedDict
from .motion_process import recover_from_ric
import torch
import clip
from exit.utils import cosine_schedule
from exit.utils import generate_src_mask
from dataset import dataset_control

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

class TextCLIP(torch.nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model
        
    def forward(self,text):
        with torch.no_grad():
            word_emb = self.model.token_embedding(text).type(self.model.dtype)
            word_emb = word_emb + self.model.positional_embedding.type(self.model.dtype)
            word_emb = word_emb.permute(1, 0, 2)  # NLD -> LND
            word_emb = self.model.transformer(word_emb)
            word_emb = self.model.ln_final(word_emb).permute(1, 0, 2).float()
            enctxt = self.model.encode_text(text).float()
        return enctxt, word_emb

def evaluate_control(motion_loader, net, mean, std, total_iter=1):

    l2_dict = OrderedDict({})
    skating_ratio_dict = OrderedDict({})
    trajectory_score_dict = OrderedDict({})

    print('========== Evaluating Control ==========')
    # all_dist = []
    all_size = 0
    dist_sum = 0
    skate_ratio_sum = 0
    traj_err = []
    traj_err_key = traj_err_key = ["traj_fail_20cm", "traj_fail_50cm", "kps_fail_20cm", "kps_fail_50cm", "kps_mean_err(m)"]
    motion_loader_name = 'aa'

    ##### ---- CLIP ---- #####
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
    clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_model = TextCLIP(clip_model)

    with torch.no_grad():
        for idx, batch in enumerate(motion_loader):
            word_embeddings, pos_one_hots, clip_text, sent_len, gt_motion, real_length, txt_tokens, traj, traj_mask_263, traj_mask = batch
            b, max_length, num_features = gt_motion.shape
            gt_motion = gt_motion.cuda()
            real_length = real_length.cuda()
            # if real_length > 150:
            #     continue 
            traj = traj.cuda()
            traj_mask_263 = traj_mask_263.cuda()
            traj_mask = traj_mask.cuda()
            partial_emb = gt_motion * traj_mask_263
            
            

            # forward
            text = clip.tokenize(clip_text, truncate=True).cuda()        
            text_emb, w_emb = clip_model(text) # (b,512) 

            pred_motion = torch.zeros(gt_motion.shape).to(gt_motion.device)

            for nb_iter in range(1, total_iter + 1):
                # partial_emb: (1,196,263)，在时间维度上在实际长度以外的会强制置为全0，导致输出的动作在实际长度以外的就会漂移
                # 可改进：仅在实际长度内保留partial_emb的先验信息，在实际长度外的，即网络生成的，依然执行随机mask，这样可以实现任意长动作生成
                pred_motion = torch.where(partial_emb == 0,pred_motion,partial_emb) # 
                # 以下为修改后
                # real_mask = generate_src_mask(target_length, real_length)
                # pred_motion = torch.where(partial_emb == 0,pred_motion,partial_emb) 

                pred_motion = net(pred_motion, text_emb, w_emb).detach() #autograd memory leak, crash with high iter number

                timestep = torch.clamp(torch.tensor(nb_iter/(total_iter)), max=1)
                ratio = cosine_schedule(timestep)
                if nb_iter != total_iter:
                    pred_motion = random_mask_motion(pred_motion,traj_mask_263,ratio)
            
            motions = pred_motion

            motions = motions * std + mean
            motions = motions.float()
            n_joints = 22 if motions.shape[-1] == 263 else 21
            motions = recover_from_ric(motions, n_joints)
            if n_joints == 21:
                motions = motions * 0.001
            
            # foot skating error
            if n_joints == 21:
                skate_ratio, skate_vel = calculate_skating_ratio_kit(motions.permute(0, 2, 3, 1))  # [batch_size]
            else:
                skate_ratio, skate_vel = calculate_skating_ratio(motions.permute(0, 2, 3, 1))  # [batch_size]
            skate_ratio_sum += skate_ratio.sum()

            # control l2 error
            # process hint
            if n_joints == 21:
                traj = traj * 0.001
            for motion, h, mask in zip(motions.cpu(), traj.cpu(), traj_mask.cpu()):
                control_error = control_l2(motion.unsqueeze(0).numpy(), h.unsqueeze(0).numpy(), mask.unsqueeze(0).numpy())
                mean_error = control_error.sum() / mask.sum()
                dist_sum += mean_error
                control_error = control_error.reshape(-1)
                mask = mask.reshape(-1)
                err_np = calculate_trajectory_error(control_error, mean_error, mask)
                
                traj_err.append(err_np)

            all_size += motions.shape[0]

        # l2 dist
        dist_mean = dist_sum / all_size
        l2_dict[motion_loader_name] = dist_mean

        # Skating evaluation
        skating_score = skate_ratio_sum / all_size
        skating_ratio_dict[motion_loader_name] = skating_score

        ### For trajecotry evaluation from GMD ###
        traj_err = np.stack(traj_err).mean(0)
        trajectory_score_dict[motion_loader_name] = traj_err

    print(f'---> [{motion_loader_name}] Control L2 dist: {dist_mean:.4f}')
    print(f'---> [{motion_loader_name}] Skating Ratio: {skating_score:.4f}')
    line = f'---> [{motion_loader_name}] Trajectory Error: '
    for (k, v) in zip(traj_err_key, traj_err):
        line += '(%s): %.4f ' % (k, np.mean(v))
    print(line)
    return l2_dict, skating_ratio_dict, trajectory_score_dict

def evaluate_control_whentraining(test_loader, clip_model, diffusion, mean, std, args, logger, batch_size=32):
    

    l2_dict = OrderedDict({})
    skating_ratio_dict = OrderedDict({})
    trajectory_score_dict = OrderedDict({})

    print('========== Evaluating Control ==========')
    # all_dist = []
    all_size = 0
    dist_sum = 0
    skate_ratio_sum = 0
    traj_err = []
    traj_err_key = traj_err_key = ["traj_fail_20cm", "traj_fail_50cm", "kps_fail_20cm", "kps_fail_50cm", "kps_mean_err(m)"]
    motion_loader_name = 'aa'


    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
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
            condition['traj'] = traj
            condition['text_emb'] = text_emb
            condition['word_emb'] = word_emb
            condition['traj_mask'] = traj_mask
            condition['traj_mask_263'] = traj_mask_263
            condition['gt_motion'] = gt_motion
            condition['real_mask'] = real_mask
            condition['clip_text'] = clip_text

            # # 采样根节点轨迹
            if diffusion.modeltype == 'omni67': 
                pred_ric = diffusion.p_sample_loop(partial_emb=None, model_kwargs=condition, batch_size=batch_size)

            if args.normalize_traj:
                traj = traj * diffusion.raw_std + diffusion.raw_mean

            
            motions = pred_ric
            num_features = motions.shape[-1]
            # motions = motions * mean_for_eval[..., :num_features] + std_for_eval[..., :num_features]
            motions = motions * std[..., :num_features] + mean[..., :num_features]
            motions = motions.float()
            n_joints = 22 if motions.shape[-1] == 67 else 21
            motions = recover_from_ric(motions, n_joints)
            if n_joints == 21:
                motions = motions * 0.001
            
            # foot skating error
            if n_joints == 21:
                skate_ratio, skate_vel = calculate_skating_ratio_kit(motions.permute(0, 2, 3, 1))  # [batch_size]
            else:
                skate_ratio, skate_vel = calculate_skating_ratio(motions.permute(0, 2, 3, 1))  # [batch_size]
            skate_ratio_sum += skate_ratio.sum()

            # control l2 error
            # process hint
            if n_joints == 21:
                traj = traj * 0.001
            for motion, h, mask in zip(motions.cpu(), traj.cpu(), traj_mask.cpu()):
                control_error = control_l2(motion.unsqueeze(0).numpy(), h.unsqueeze(0).numpy(), mask.unsqueeze(0).numpy())
                mean_error = control_error.sum() / mask.sum()
                dist_sum += mean_error
                control_error = control_error.reshape(-1)
                mask = mask.reshape(-1)
                err_np = calculate_trajectory_error(control_error, mean_error, mask)
                
                traj_err.append(err_np)
            # logger.info(f'{idx}/{len(test_loader)}, mean_error={mean_error:.4f}')
            all_size += motions.shape[0]
            break

        # l2 dist
        dist_mean = dist_sum / all_size
        l2_dict[motion_loader_name] = dist_mean

        # Skating evaluation
        skating_score = skate_ratio_sum / all_size
        skating_ratio_dict[motion_loader_name] = skating_score

        ### For trajecotry evaluation from GMD ###
        traj_err = np.stack(traj_err).mean(0)
        trajectory_score_dict[motion_loader_name] = traj_err

    line = f'---> [{motion_loader_name}] Trajectory Error: '
    for (k, v) in zip(traj_err_key, traj_err):
        line += '(%s): %.4f ' % (k, np.mean(v))
    logger.info(f'---> [{motion_loader_name}] Control L2 dist: {dist_mean:.4f}')
    logger.info(f'---> [{motion_loader_name}] Skating Ratio: {skating_score:.4f}')
    logger.info(line)
    return dist_mean, skating_ratio_dict, trajectory_score_dict

def evaluate_control_diffmae(motion_loader, diffusion_root, mean, std, args, logger, batch_size=32, diffusion=None):

    l2_dict = OrderedDict({})
    skating_ratio_dict = OrderedDict({})
    trajectory_score_dict = OrderedDict({})

    print('========== Evaluating Control ==========')
    # all_dist = []
    all_size = 0
    dist_sum = 0
    skate_ratio_sum = 0
    traj_err = []
    traj_err_key = traj_err_key = ["traj_fail_20cm", "traj_fail_50cm", "kps_fail_20cm", "kps_fail_50cm", "kps_mean_err(m)"]
    motion_loader_name = 'aa'

    ##### ---- CLIP ---- #####
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
    clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_model = TextCLIP(clip_model)

    if motion_loader.dataset.split == 'test':
        mean_for_eval = torch.from_numpy(motion_loader.dataset.mean_for_eval).cuda()[None, None, :]
        std_for_eval = torch.from_numpy(motion_loader.dataset.std_for_eval).cuda()[None, None, :]

    with torch.no_grad():
        for idx, batch in enumerate(motion_loader):
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
            condition['traj'] = traj
            condition['text_emb'] = text_emb
            condition['word_emb'] = word_emb
            condition['traj_mask'] = traj_mask
            condition['traj_mask_263'] = traj_mask_263
            condition['gt_motion'] = gt_motion
            condition['traj'] = traj
            condition['real_mask'] = real_mask
            condition['clip_text'] = clip_text
            # condition['mean'] = mean # 这里赋值，就是guide里面使用
            # condition['std'] = std

            # # 采样根节点轨迹
            if diffusion_root.modeltype == 'omni67': 
                pred_ric = diffusion_root.p_sample_loop(partial_emb=None, model_kwargs=condition, batch_size=batch_size)
                loss, msg = diffusion_root.calc_loss(gt_ric, pred_ric, mean[..., :67], std[..., :67], traj_mask, traj, real_mask, traj_mask_263, 67, 1)

            if args.normalize_traj:
                traj = traj * diffusion_root.raw_std + diffusion_root.raw_mean

            ### 加上2阶段，但是只引导第一次，做Forced Guidance的消融实验
            if diffusion != None:
                partial_emb = torch.zeros_like(gt_motion, device=gt_motion.device)
                partial_emb[..., :67] = pred_ric # 使用dataloader出来的
                # partial_emb[..., :67] = gt_ric 
                pred_motion = diffusion.p_sample_loop(partial_emb, with_control=False, model_kwargs=condition, batch_size=batch_size, control_once=True)
                motions = pred_motion
            ###
            else:
                motions = pred_ric
            num_features = motions.shape[-1]
            # motions = motions * mean_for_eval[..., :num_features] + std_for_eval[..., :num_features]
            motions = motions * std[..., :num_features] + mean[..., :num_features]
            motions = motions.float()
            n_joints = 22 # if motions.shape[-1] == 67 else 21
            motions = recover_from_ric(motions, n_joints)
            if n_joints == 21:
                motions = motions * 0.001
            
            # foot skating error
            if n_joints == 21:
                skate_ratio, skate_vel = calculate_skating_ratio_kit(motions.permute(0, 2, 3, 1))  # [batch_size]
            else:
                skate_ratio, skate_vel = calculate_skating_ratio(motions.permute(0, 2, 3, 1))  # [batch_size]
            skate_ratio_sum += skate_ratio.sum()

            # control l2 error
            # process hint
            if n_joints == 21:
                traj = traj * 0.001
            for motion, h, mask in zip(motions.cpu(), traj.cpu(), traj_mask.cpu()):
                control_error = control_l2(motion.unsqueeze(0).numpy(), h.unsqueeze(0).numpy(), mask.unsqueeze(0).numpy())
                mean_error = control_error.sum() / mask.sum()
                dist_sum += mean_error
                control_error = control_error.reshape(-1)
                mask = mask.reshape(-1)
                err_np = calculate_trajectory_error(control_error, mean_error, mask)
                
                traj_err.append(err_np)
            logger.info(f'{idx}/{len(motion_loader)}, mean_error={mean_error:.4f}')
            all_size += motions.shape[0]


        # l2 dist
        dist_mean = dist_sum / all_size
        l2_dict[motion_loader_name] = dist_mean

        # Skating evaluation
        skating_score = skate_ratio_sum / all_size
        skating_ratio_dict[motion_loader_name] = skating_score

        ### For trajecotry evaluation from GMD ###
        traj_err = np.stack(traj_err).mean(0)
        trajectory_score_dict[motion_loader_name] = traj_err

    print(f'---> [{motion_loader_name}] Control L2 dist: {dist_mean:.4f}')
    print(f'---> [{motion_loader_name}] Skating Ratio: {skating_score:.4f}')
    line = f'---> [{motion_loader_name}] Trajectory Error: '
    for (k, v) in zip(traj_err_key, traj_err):
        line += '(%s): %.4f ' % (k, np.mean(v))
    print(line)
    logger.info(f'---> [{motion_loader_name}] Control L2 dist: {dist_mean:.4f}')
    logger.info(f'---> [{motion_loader_name}] Skating Ratio: {skating_score:.4f}')
    logger.info(line)
    return l2_dict, skating_ratio_dict, trajectory_score_dict

# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists

def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat) # mat每一行都是0-31数值的索引，该行从左到右的含义是该数值的索引所对应原矩阵的值按升序排列。如果mat是一个完美的匹配矩阵，bool_mat[0,:]应该是从0到31的，即第i行对应的最小匹配数值就是原矩阵中的该行第i列即[i,i]
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i]) # 对于bool_mat，如果每行的True出现在第0个位置，说明第0个就是正确值；如果True出现在第1个位置，说明第1个就是正确值。所以for循环先从[:,0]查找True，一直到[:,2]
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1) # 是个(32,3)的矩阵，每一行可以视为一个从左到右的阶跃函数，即如果该行第0个就对了，那么3个值都是T；如果第1个才对，那么3个值是F,T,T；如果第2个才对，那么3个值是F,F,T；如果都错了，那3个值是F,F,F,
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0)
    else:
        return top_k_mat


def calculate_matching_score(embedding1, embedding2, sum_all=False):
    assert len(embedding1.shape) == 2
    assert embedding1.shape[0] == embedding2.shape[0]
    assert embedding1.shape[1] == embedding2.shape[1]

    dist = linalg.norm(embedding1 - embedding2, axis=1)
    if sum_all:
        return dist.sum(axis=0)
    else:
        return dist



def calculate_activation_statistics(activations):
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative dataset set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative dataset set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_trajectory_error(dist_error, mean_err_traj, mask, strict=True):
    ''' dist_error shape [5]: error for each kps in metre
      Two threshold: 20 cm and 50 cm.
    If mean error in sequence is more then the threshold, fails
    return: traj_fail(0.2), traj_fail(0.5), all_kps_fail(0.2), all_kps_fail(0.5), all_mean_err.
        Every metrics are already averaged.
    '''
    # mean_err_traj = dist_error.mean(1)
    if strict:
        # Traj fails if any of the key frame fails
        traj_fail_02 = 1.0 - (dist_error <= 0.2).all()
        traj_fail_05 = 1.0 - (dist_error <= 0.5).all()
    else:
        # Traj fails if the mean error of all keyframes more than the threshold
        traj_fail_02 = (mean_err_traj > 0.2)
        traj_fail_05 = (mean_err_traj > 0.5)
    all_fail_02 = (dist_error > 0.2).sum() / mask.sum()
    all_fail_05 = (dist_error > 0.5).sum() / mask.sum()

    # out = {"traj_fail_02": traj_fail_02,
    #        "traj_fail_05": traj_fail_05,
    #        "all_fail_02": all_fail_02,
    #        "all_fail_05": all_fail_05,
    #        "all_mean_err": dist_error.mean()}
    return np.array([traj_fail_02, traj_fail_05, all_fail_02, all_fail_05, dist_error.sum() / mask.sum()])


def calculate_trajectory_diversity(trajectories, lengths):
    ''' Standard diviation of point locations in the trajectories
    Args:
        trajectories: [bs, rep, 196, 2]
        lengths: [bs]
    '''
    # [32, 2, 196, 2 (xz)]
    # mean_trajs = trajectories.mean(1, keepdims=True)
    # dist_to_mean = np.linalg.norm(trajectories - mean_trajs, axis=3)
    def traj_div(traj, length):
        # traj [rep, 196, 2]
        # length (int)
        traj = traj[:, :length, :]
        # point_var = traj.var(axis=0, keepdims=True).mean()
        # point_var = np.sqrt(point_var)
        # return point_var

        mean_traj = traj.mean(axis=0, keepdims=True)
        dist = np.sqrt(((traj - mean_traj)**2).sum(axis=2))
        rms_dist = np.sqrt((dist**2).mean())
        return rms_dist
        
    div = []
    for i in range(len(trajectories)):
        div.append(traj_div(trajectories[i], lengths[i]))
    return np.array(div).mean()


def calculate_skating_ratio(motions):
    thresh_height = 0.05 # 10
    fps = 20.0
    thresh_vel = 0.50 # 20 cm /s 
    avg_window = 5 # frames

    batch_size = motions.shape[0]
    # 10 left, 11 right foot. XZ plane, y up
    # motions [bs, 22, 3, max_len]
    verts_feet = motions[:, [10, 11], :, :].detach().cpu().numpy()  # [bs, 2, 3, max_len]
    verts_feet_plane_vel = np.linalg.norm(verts_feet[:, :, [0, 2], 1:] - verts_feet[:, :, [0, 2], :-1],  axis=2) * fps  # [bs, 2, max_len-1]
    # [bs, 2, max_len-1]
    vel_avg = uniform_filter1d(verts_feet_plane_vel, axis=-1, size=avg_window, mode='constant', origin=0)

    verts_feet_height = verts_feet[:, :, 1, :]  # [bs, 2, max_len]
    # If feet touch ground in agjecent frames
    feet_contact = np.logical_and((verts_feet_height[:, :, :-1] < thresh_height), (verts_feet_height[:, :, 1:] < thresh_height))  # [bs, 2, max_len - 1]
    # skate velocity
    skate_vel = feet_contact * vel_avg

    # it must both skating in the current frame
    skating = np.logical_and(feet_contact, (verts_feet_plane_vel > thresh_vel))
    # and also skate in the windows of frames
    skating = np.logical_and(skating, (vel_avg > thresh_vel))

    # Both feet slide
    skating = np.logical_or(skating[:, 0, :], skating[:, 1, :]) # [bs, max_len -1]
    skating_ratio = np.sum(skating, axis=1) / skating.shape[1]
    
    return skating_ratio, skate_vel
    
    # verts_feet_gt = markers_got[:, [16, 47], :].detach().cpu().numpy() # [119, 2, 3] heels
    # verts_feet_horizon_vel_gt = np.linalg.norm(verts_feet_gt[1:, :, :-1] - verts_feet_gt[:-1, :, :-1],  axis=-1) * 30
    
    # verts_feet_height_gt = verts_feet_gt[:, :, -1][0:-1] # [118,2]
    # min_z = markers_gt[:, :, 2].min().detach().cpu().numpy()
    # verts_feet_height_gt  = verts_feet_height_gt - min_z

    # skating_gt = (verts_feet_horizon_vel_gt > thresh_vel) * (verts_feet_height_gt < thresh_height)
    # skating_gt = np.sum(np.logival_and(skating_gt[:, 0], skating_gt[:, 1])) / 118
    # skating_gt_list.append(skating_gt)


def calculate_skating_ratio_kit(motions):
    thresh_height = 0.05 # 10
    fps = 20.0
    thresh_vel = 0.50 # 20 cm /s 
    avg_window = 5 # frames

    batch_size = motions.shape[0]
    # 15 left, 20 right foot. XZ plane, y up
    # motions [bs, 22, 3, max_len]
    verts_feet = motions[:, [15, 20], :, :].detach().cpu().numpy()  # [bs, 2, 3, max_len]
    verts_feet_plane_vel = np.linalg.norm(verts_feet[:, :, [0, 2], 1:] - verts_feet[:, :, [0, 2], :-1],  axis=2) * fps  # [bs, 2, max_len-1]
    # [bs, 2, max_len-1]
    vel_avg = uniform_filter1d(verts_feet_plane_vel, axis=-1, size=avg_window, mode='constant', origin=0)

    verts_feet_height = verts_feet[:, :, 1, :]  # [bs, 2, max_len]
    # If feet touch ground in agjecent frames
    feet_contact = np.logical_and((verts_feet_height[:, :, :-1] < thresh_height), (verts_feet_height[:, :, 1:] < thresh_height))  # [bs, 2, max_len - 1]
    # skate velocity
    skate_vel = feet_contact * vel_avg

    # it must both skating in the current frame
    skating = np.logical_and(feet_contact, (verts_feet_plane_vel > thresh_vel))
    # and also skate in the windows of frames
    skating = np.logical_and(skating, (vel_avg > thresh_vel))

    # Both feet slide
    skating = np.logical_or(skating[:, 0, :], skating[:, 1, :]) # [bs, max_len -1]
    skating_ratio = np.sum(skating, axis=1) / skating.shape[1]
    
    return skating_ratio, skate_vel


def control_l2(motion, hint, hint_mask):
    # motion: b, seq, 22, 3
    # hint: b, seq, 22, 1
    loss = np.linalg.norm((motion - hint) * hint_mask, axis=-1)
    # loss = loss.sum() / hint_mask.sum()
    return loss

def cross_combination_joints():
    controllable_joints = {
        "pelvis": 0,
        "l_foot": 10,
        "r_foot": 11,
        "head": 15,
        "left_wrist": 20,
        "right_wrist": 21,
    }
    choose_combination = [
        [0],
        [10],
        [11],
        [15],
        [20],
        [21],
        [0, 10],
        [0, 11],
        [0, 15],
        [0, 20],
        [0, 21],
        [10, 11],
        [10, 15],
        [10, 20],
        [10, 21],
        [11, 15],
        [11, 20],
        [11, 21],
        [15, 20],
        [15, 21],
        [20, 21],
        [0, 10, 11],
        [0, 10, 15],
        [0, 10, 20],
        [0, 10, 21],
        [0, 11, 15],
        [0, 11, 20],
        [0, 11, 21],
        [0, 15, 20],
        [0, 15, 21],
        [0, 20, 21],
        [10, 11, 15],
        [10, 11, 20],
        [10, 11, 21],
        [10, 15, 20],
        [10, 15, 21],
        [10, 20, 21],
        [11, 15, 20],
        [11, 15, 21],
        [11, 20, 21],
        [15, 20, 21],
        [0, 10, 11, 15],
        [0, 10, 11, 20],
        [0, 10, 11, 21],
        [0, 10, 15, 20],
        [0, 10, 15, 21],
        [0, 10, 20, 21],
        [0, 11, 15, 20],
        [0, 11, 15, 21],
        [0, 11, 20, 21],
        [0, 15, 20, 21],
        [10, 11, 15, 20],
        [10, 11, 15, 21],
        [10, 11, 20, 21],
        [10, 15, 20, 21],
        [11, 15, 20, 21],
        [0, 10, 11, 15, 20],
        [0, 10, 11, 15, 21],
        [0, 10, 11, 20, 21],
        [0, 10, 15, 20, 21],
        [0, 11, 15, 20, 21],
        [10, 11, 15, 20, 21],
        [0, 10, 11, 15, 20, 21],
    ]
    return choose_combination
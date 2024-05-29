import torch
import torch.nn.functional as F
import numpy as np
from utils.motion_process import recover_from_ric, recover_root_rot_pos
from exit.utils import generate_src_mask, top_k, visualize_2motions
from einops import rearrange

def random_mask_motion(args, motion, traj_mask_263, real_length):
    '''
    保留motion中与控制关节点相关的值; 将非控制关节点部分的随机mask置0
    motion: (b,196.263)
    traj_mask_263: (b,196,263,)
    real_length: (b)
    '''
    B, L, num_features = motion.shape
    mask_min = args.mask_range[0]
    mask_max = args.mask_range[1]
    ### temporal mask  TODO: 时间mask还不严谨，应该取真是长度来算mask的个数，但是创建mask的时候，只能在有效长度内执行mask，有效长度外的必须为False
    rand_mask_probs = torch.zeros((B, )).uniform_(mask_min, mask_max).to(real_length.device) # 随机mask比例
    num_token_masked = (real_length * rand_mask_probs).round().clamp(min=1)  # (b,)
    # 下面两行目的就是按照生成的motion mask百分比，生成随机的mask。实现思路是：生成一串随机数，通过排序索引，然后将索引小于要被mask数量的位置就变成True，以此实现随机mask生成
    batch_randperm = torch.rand((B, L)).argsort(dim=-1).cuda()
    temporal_mask = ~(batch_randperm < num_token_masked.unsqueeze(-1)) # 要被mask的位置是False  (b,L)

    ### spatial mask
    rand_mask_probs = torch.zeros((B, )).uniform_(0.5,1).to(real_length.device) # 随机mask比例
    num_token_masked = (torch.tensor([num_features]*B).to(rand_mask_probs.device) * rand_mask_probs).round().clamp(min=1)
    batch_randperm = torch.rand((B, num_features)).argsort(dim=-1).cuda()
    spatial_mask = ~(batch_randperm < num_token_masked.unsqueeze(-1)) # 要被mask的位置是False  (b,263)
    if np.random.uniform() < args.mask_root:
        spatial_mask[:, :4] = False
    if np.random.uniform() < args.mask_vel:
        spatial_mask[:, 193:259] = False

    mask = (temporal_mask[...,None] & spatial_mask[:,None]) | traj_mask_263
    masked_motion = motion * mask
    return masked_motion.float()

def random_mask_motion2(args, motion, traj_mask_263, real_length, ratio=None):
    '''
    保留motion中与控制关节点相关的值; 将非控制关节点部分的随机mask置0
    motion: (b,196.263)
    traj_mask_263: (b,196,263,)
    real_length: (b)
    '''
    B, L, num_features = motion.shape
    mask_min = args.mask_range[0]
    mask_max = args.mask_range[1]
    ### temporal mask 
    if ratio is not None:
        thr = torch.zeros(B,1).uniform_(ratio, ratio).clamp(max=1)
    else:
        thr = torch.zeros(B,1).uniform_(0.5,1.0).clamp(max=1)
    batch_mask = (torch.zeros(B,L).uniform_(0,1)>thr).to(real_length.device)
    length_mask = (torch.arange(L).expand(B,L).to(real_length.device)) < real_length[...,None]
    temporal_mask = batch_mask & length_mask # 要被mask的位置是False  (b,L)

    ### spatial mask
    rand_mask_probs = torch.zeros((B, )).uniform_(0.5,1).to(real_length.device) # 随机mask比例
    num_token_masked = (torch.tensor([num_features]*B).to(rand_mask_probs.device) * rand_mask_probs).round().clamp(min=1)
    batch_randperm = torch.rand((B, num_features)).argsort(dim=-1).cuda()
    spatial_mask = ~(batch_randperm < num_token_masked.unsqueeze(-1)) # 要被mask的位置是False  (b,263)
    if np.random.uniform() < args.mask_root:
        spatial_mask[:, :4] = False
    if np.random.uniform() < args.mask_vel:
        spatial_mask[:, 193:259] = False

    mask = (temporal_mask[...,None] & spatial_mask[:,None]) | traj_mask_263 # (B,L, dim)
    masked_motion = motion * mask

    noise = torch.randn_like(masked_motion, device=masked_motion.device) # (B,L, dim)
    if args.mask_noise: # 仅对mask后值为0的地方加噪
        masked_motion += noise * ~mask
    if args.root_mask_noise_all: # 所有值都加噪声
        # 对保留的值，以0.5的概率加噪
        mask = mask * (torch.zeros(B,L, num_features, device=mask.device).uniform_(0,1) < 0.5) 
        masked_motion += noise * ~mask
    return masked_motion.float()

def random_window_mask(motion, real_length):
    '''
    为了加入时间上的补全功能。随机选取窗口比例0~1, 根据实际动作长度计算窗口长度, 随机选取窗口起始点idx, 将窗口部分的动作mask掉
    这里不会对motion的263向量进行随机mask
    '''
    B, L, num_features = motion.shape
    window_prob = torch.zeros((B,)).uniform_(0.2, 0.5).to(real_length.device) # 窗口比例0.2~0.5
    window_size = window_prob * real_length # 窗口长度
    low_bound = torch.rand((B,)).to(real_length.device) * (real_length - window_size) # 窗口下界
    high_bound = low_bound + window_size # 窗口上界
    batch_randperm = torch.arange(0,L).repeat(B,1).to(real_length.device)
    # 生成mask
    completion_mask = ~(torch.logical_and(batch_randperm >= low_bound[..., None], batch_randperm <= high_bound[..., None]))
    masked_motion = motion * completion_mask[..., None]
    return masked_motion

def random_mask_root(args, gt_root, real_length):
    '''
    gt_root: (b,196,4)
    real_length: (b,

    return : (b,196,4)
    '''
    B, L, num_features = gt_root.shape
    mask_min = args.mask_range[0]
    mask_max = args.mask_range[1]
    rand_mask_probs = torch.zeros((B, )).uniform_(mask_min, mask_max).to(real_length.device) # 随机mask比例
    num_token_masked = (real_length * rand_mask_probs).round().clamp(min=1)  # (b,)
    batch_randperm = torch.rand((B, L)).argsort(dim=-1).cuda()
    temporal_mask = ~(batch_randperm < num_token_masked.unsqueeze(-1)) # 要被mask的位置是False  (b,L)
    
    real_mask = generate_src_mask(L, real_length)
    mask = (temporal_mask & real_mask)[..., None] # 与real_mask做与运算，将有效长度以外的必定mask
    masked_root = gt_root * mask
    return masked_root.float()

def random_mask_root_prob(args, gt_root,real_length):
    '''random mask batch root data in random ratio in 0.5->1
    gt_root: (b,196,4)
    real_length: (b,)
    return : (b,196,4)
    '''
    B, L, num_features = gt_root.shape
    thr = torch.zeros(B,1).uniform_(0.3,1.0).clamp(max=1) # 这个是mask的比例，不是保留的比例
    batch_mask = (torch.zeros(B,L).uniform_(0,1)>thr).to(real_length.device) # 实际长度的mask
    length_mask = (torch.arange(L).expand(B,L).to(real_length.device)) < real_length[...,None] # 有效长度的mask 
    batch_mask = batch_mask & length_mask # 要mask的为False
    masked_root = gt_root * batch_mask[...,None] # mask后的值为0
    if args.mask_noise: # 仅对mask后值为0的地方加噪
        noise = torch.randn_like(masked_root, device=masked_root.device)
        masked_root += noise * ~batch_mask[...,None]
    if args.root_mask_noise_all: # 所有值都加噪声
        noise = torch.randn_like(masked_root, device=masked_root.device)
        # 对保留的值，以0.5的概率加噪
        mask = batch_mask * (torch.zeros(B,L, device=batch_mask.device).uniform_(0,1) < 0.5) 
        masked_root += noise * ~mask[...,None]
    return masked_root.float()

def random_mask_token(token, motion_token_len, args):
    assert motion_token_len.max() <= 49
    batch_size, max_token_len = token.shape
    mask = torch.bernoulli(args.pkeep * torch.ones(token.shape,
                                                device=token.device))
    # 仅对motion toke做mask. To prevent pad token got mixed up.
    real_mask_no_end = generate_src_mask(max_token_len, motion_token_len) # 表示原动作的mask   11111100
    mask = torch.logical_or(mask, ~real_mask_no_end).int()
    r_indices = torch.randint_like(token, args.nb_code) #  TODO  这是什么？  随机替换为其他idx？
    input_indices = mask*token+(1-mask)*r_indices

    # 时间上的mask
    mask_id = args.nb_code + 2 #  8194
    rand_mask_probs = torch.zeros(batch_size, device = motion_token_len.device).float().uniform_(0.5, 1) #  (b,）
    num_token_masked = (motion_token_len * rand_mask_probs).round().clamp(min = 1) # 每个batch mask的数量(b,)
    real_mask = generate_src_mask(max_token_len, motion_token_len+1) # 表示原动作的mask（包含end token）   11111100
    batch_randperm = torch.rand((batch_size, max_token_len), device = token.device) - real_mask_no_end.int()
    batch_randperm = batch_randperm.argsort(dim = -1)
    token_mask = batch_randperm < rearrange(num_token_masked, 'b -> b 1') # 要mask的是True

    masked_input_indices = torch.where(token_mask, mask_id, input_indices) # 网络的输入，包含mask token和被保留的输入token

    return masked_input_indices, real_mask, real_mask_no_end, token_mask

def gradients(x, gt, hint, mask_hint, mean, std):
    with torch.enable_grad():
        x.requires_grad_(True)

        x_ = x * std + mean
        # n_joints = 22 if x_.shape[-1] == 263 else 21
        joint_pos = recover_from_ric(x_, joints_num=22) # 全局xyz
        # if n_joints == 21: # 猜测是KIT格式, 就要把 毫米转为米？
        #     joint_pos = joint_pos * 0.001
        #     hint = hint * 0.001
        try:
            assert torch.allclose(hint.nonzero(),mask_hint.nonzero())
        except:
            a = 1
        loss = torch.norm((joint_pos - hint) * mask_hint, dim=-1, p=2)
        l1 = F.l1_loss(joint_pos * mask_hint, hint * mask_hint, reduction='none') # (b,196,22,3)
        l2 = F.mse_loss(joint_pos * mask_hint, hint * mask_hint, reduction='none')
        assert torch.allclose(loss, l2.sum(-1).sqrt())

        grad = torch.autograd.grad([loss.sum()], [x])[0] # (b,196,67)
        ####  debug
        l = loss[loss!=0].sort()
        # control_joint = list(set(mask_hint.nonzero()[:,2].tolist()))[0]
        # frames = set(mask_hint.nonzero()[:,1].tolist())
        # idx = set(grad.nonzero()[:,2].tolist())
        # a1 = hint[0,50,10,:]
        # a2 = joint_pos[0,50,10,:] 
        # ((a1-a2)**2).sum().sqrt()
        # print('a1=',a1)
        # print('a2=',a2)
        # print('loss = ', loss[0,50,10])
        # print('gt = ', gt[0,50,:4])
        # print('x =', x[0,50,:4])
        # print('grad = ', grad[0,50,:4])
        ####
        grad = top_k(grad)
        # the motion in HumanML3D always starts at the origin (0,y,0), so we zero out the gradients for the root joint
        # grad[..., 0] = 0
        x.detach()    
    x = x - 0.1 * grad
    # print('gt = ', gt[0,50,:4])
    # print('x =', x[0,50,:4])
    return loss, grad, x

def calc_grad_scale(mask_hint):
    assert mask_hint.shape[1] == 196
    num_keyframes = mask_hint.sum(dim=1).squeeze(-1)
    max_keyframes = num_keyframes.max(dim=1)[0]
    scale = 20 / max_keyframes
    return scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

def top_k(grad, threshold=1e-2):
    '''
    将grad中, 梯度小于一定阈值的保留下来，其他的继续更新 
    grad: 
    '''
    threshold_mask = grad.abs() > threshold 
    non_zero_mask = grad.abs()!=0 
    # print('num = ', non_zero_mask.sum())
    # print('min=', grad.abs()[non_zero_mask].min())
    return grad * non_zero_mask

def root_dist_loss(gt_root, pred_root, real_mask, args):
    gt_rot, gt_pos = recover_root_rot_pos(gt_root)
    pred_rot, pred_pos = recover_root_rot_pos(pred_root)
    if args.loss_type == 'l1':
        loss_rotate = F.l1_loss(gt_rot[real_mask], pred_rot[real_mask], reduction='mean')
        loss_position = F.l1_loss(gt_pos[real_mask], pred_pos[real_mask], reduction='mean')
    elif args.loss_type == 'l2':
        loss_rotate = F.mse_loss(gt_rot[real_mask], pred_rot[real_mask], reduction='mean')
        loss_position = F.mse_loss(gt_pos[real_mask], pred_pos[real_mask], reduction='mean')
    return loss_rotate, loss_position, gt_pos, pred_pos

def recover(motion):
    mean = np.load('dataset/HumanML3D/Mean.npy')[None, ...] # dataset/HumanML3D/Mean.npy
    std = np.load('dataset/HumanML3D/Std.npy')[None, ...]
    humanml_mean = torch.from_numpy(mean)[None, ...].cuda()
    humanml_std = torch.from_numpy(std)[None, ...].cuda()
    xyz = recover_from_ric(motion * humanml_std + humanml_mean, joints_num=22)
    return xyz

def vis_motion(pred, gt=None):
    mean = np.load('dataset/HumanML3D/Mean.npy')[None, ...] # dataset/HumanML3D/Mean.npy
    std = np.load('dataset/HumanML3D/Std.npy')[None, ...]
    visualize_2motions(pred[0].detach().cpu().numpy(), std, mean, 't2m', None, motion2=None if gt==None else gt[0].detach().cpu().numpy(), save_path='./output/testsample/1.html')

def complete_mask(traj_mask_263, traj_mask):
    control_id = traj_mask[0].sum(0).sum(-1).nonzero()
    traj_mask_263[..., :4] = True
    traj_mask_263[..., 4+3*(control_id-1):4+3*control_id] = True # ric  21*3
    return traj_mask_263

def calc_loss_xyz(traj_mask, gt, pred, mean, std):
    recon_xyz = recover_from_ric(pred * std + mean, joints_num=22)  # 反归一化再转全局xyz
    gt_xyz = recover_from_ric(gt * std + mean, joints_num=22)
    loss = F.l1_loss(recon_xyz[traj_mask], gt_xyz[traj_mask])
    return loss

def create_trajmask263(joint_id):
    batchsize = joint_id.shape[0]
    traj_mask_263 = torch.zeros(b, 196, 263)
    for b in range(batchsize):
        traj_mask_263[b, :, :4] = True # root
        for i in joint_id:
            traj_mask_263[b, :, 4+3*(i-1):4+3*i] = True # ric  21*3
    return traj_mask_263

def load_ckpt(net, path):
    ckpt = torch.load(path, map_location='cpu')
    if 'module' in list(ckpt['trans'].keys())[0]:
        new_ckpt = {}
        for k, v in ckpt['trans'].items():
            new_k = k.replace('module.', '') if 'module' in k else k
            new_ckpt[new_k] = v
        net.load_state_dict(new_ckpt, strict=True)
    else:
        net.load_state_dict(ckpt['trans'], strict=True)

def generate_src_mask(T, length):
    B = len(length)
    mask = torch.arange(T).repeat(B, 1).to(length.device) < length.unsqueeze(-1)
    return mask

##### ---- CLIP ---- #####
# https://github.com/openai/CLIP/issues/111
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
    

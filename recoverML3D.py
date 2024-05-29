import torch
import numpy as np
from dataset import dataset_control

import warnings
warnings.filterwarnings('ignore')

from utils.motion_process import recover_root_rot_pos
from utils.quaternion import *

def get_ML3D_emb(root_data,traj,MLmean=None,MLstd=None,mean_raw=None,std_raw=None,use_hardcode_norm=False):
    '''restore ML3D representation from root data and control traj
    
    维度还没考虑batch dim,有些shape[0]的地方还得看
    root_data: predict root data (length*4)
    traj: global xyz of control joints (length*66)
    mean/std: value for de/normalization
    use_hardcode_norm: bool
    '''
    
    #norm value
    if use_hardcode_norm:
        mean = torch.from_numpy(np.load('dataset/HumanML3D/Mean.npy')[None, ...]).to(traj.device) # dataset/HumanML3D/Mean.npy
        std = torch.from_numpy(np.load('dataset/HumanML3D/Std.npy')[None, ...]).to(traj.device)
        spatial_mean = torch.from_numpy(np.load('dataset/humanml_spatial_norm/Mean_raw.npy').reshape(-1,3)).to(traj.device)
        spatial_std = torch.from_numpy(np.load('dataset/humanml_spatial_norm/Std_raw.npy').reshape(-1,3)).to(traj.device)
    else:
        mean = MLmean
        std = MLstd
        spatial_mean = mean_raw
        spatial_std = std_raw

    #unfold 66->22*3
    traj_unfold = traj.reshape(traj.shape[0],-1,3)
    
    #spatial denorm
    traj_denorm = (traj_unfold * spatial_std) + spatial_mean

    #算ric时要去除root
    traj_rm_root_denorm = traj_denorm.clone()[:,1:,:]
    #用0值表示mask，所以需要用反归一化前的值进行判断
    traj_rm_root = traj_unfold.clone()[:,1:,:]

    #判断控制mask
    traj_valid = torch.tensor(traj_rm_root.sum(-1)!=0)   #length*21    
    
    #合成各区间mask
    
    triple_mask = torch.repeat_interleave(traj_valid,3,-1) #length,63
    #ric length*63
    ric_mask = triple_mask
    #vel length*66
    root_vel_mask = torch.ones(triple_mask.shape[0],3).to(traj.device)
    vel_mask = torch.cat((root_vel_mask,triple_mask),-1)

    #mask vel
    #vel_mask = torch.zeros_like(vel_mask)

    #root/foot/rot
    root_mask = torch.ones_like(root_data) #暂定zero
    foot_mask = torch.zeros_like(root_data)
    rot_mask = torch.zeros_like(triple_mask)
    rot_mask = torch.cat((rot_mask,rot_mask),-1)
    #263 mask
    emb_mask = torch.cat((root_mask,ric_mask,rot_mask,vel_mask,foot_mask),-1)#length,263

    #计算emb：
    mean_root = mean[:,0:4]
    std_root = std[:,0:4]
    root_data = (root_data*std_root)+mean_root

    r_rot,r_pos = recover_root_rot_pos(root_data)#length*4, length*3

    #root xz 
    traj_rm_root_denorm[:,:,0] -= r_pos[:,0,None]
    traj_rm_root_denorm[:,:,2] -= r_pos[:,2,None]

    #ric
    ric = qrot(r_rot.unsqueeze(-2).repeat(1,21,1),traj_rm_root_denorm) #(length,21,4   length,21,3)
    ric = ric.reshape(ric.shape[0],-1)#length,63
    #vel,第一个root使用root data计算
    vel = torch.zeros((196,66)).to(traj.device)
    # vel = traj_denorm[1:] - traj_denorm[:-1]#length-1,22,3
    # vel_root = r_pos[1:] - r_pos[:-1]
    # vel[:,0,:] = vel_root
    # vel = torch.cat((vel,torch.zeros(1,22,3).to(vel.device)),0)
    # vel = qrot(r_rot.unsqueeze(-2).repeat(1,22,1),vel)
    # vel = vel.reshape(vel.shape[0],-1)
    #263 emb
    emb = torch.cat((root_data,ric,rot_mask,vel,foot_mask),-1)
    emb = (emb-mean)/std
    emb = emb*emb_mask

    return emb,emb_mask,vel

if __name__ == '__main__':
    data_loader = dataset_control.DataLoader(batch_size=1,mode='train')
    data_loader_iter = dataset_control.cycle(data_loader)

    batch = next(data_loader_iter)
    word_embeddings, pos_one_hots, clip_text, sent_len, gt_motion, real_length, txt_tokens, traj, traj_mask_263, traj_mask = batch

    print(real_length)#只在real length内符合

    gt_motion = gt_motion.squeeze(0)
    root_data = gt_motion[:,0:4]
    traj = traj.squeeze(0)

    emb,emb_mask,triplemask = get_ML3D_emb(root_data,traj,use_hardcode_norm=True)

    print(np.allclose(emb[:103],(gt_motion*emb_mask)[:103],atol=1e-5))
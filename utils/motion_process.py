import torch
from utils.quaternion import quaternion_to_cont6d, qrot, qinv

def global_to_local_xzy(r_pos, r_rot_quat):
    '''
    r_pos: (b,196,3) x y z  初始的x z一定是0, 即将人放在y轴上, 处于xz的原点
    data: (b,196,3)  x z y
    '''
    data = torch.zeros_like(r_pos).to(r_pos.device)
    # r_pos逐差计算相邻的xz差值
    data[:, :-1, :] = r_pos[:, 1:, :] - r_pos[:, 0:-1, :] # 只有195个
    data[:, -1, :] = data[:, -2, :] # 最后一个直接复制
    data_ = qrot(qinv(r_rot_quat), data) # x y z
    data[..., 0] = data_[..., 0]
    data[..., 1] = data_[..., 2]
    data[..., 2] = r_pos[..., 1] # 每个时刻根节点的y值即高度
    return data


def recover_root_rot_pos(data):
    rot_vel = data[..., 0] # (b,196) 根节点旋转角速度,第i个角速度表示第i帧向第i+1帧转的角速度
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1] # 所以这里r_rot_ang的第0帧是第0帧的真实朝向，为0，即已经向z轴了
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1) # 通过累加，得到的是每个时刻根节点相对于初始根节点即z轴的旋转角度

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device) # 获得每个时刻根节点的旋转的四元数
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3] # 每个时刻根节点的xz速度，即相邻的xz差值
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos) # 乘上对应的旋转，得到的是真正的相邻根节点的xz差值

    r_pos = torch.cumsum(r_pos, dim=-2) # 通过累加得到根节点的绝对位置

    r_pos[..., 1] = data[..., 3] # 每个时刻根节点的y值即高度

    # data_ = global_to_local_xzy(r_pos, r_rot_quat)
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4] # 这22*3个就是 ric
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions
    
import numpy as np

# raw_motion = np.load('dataset/HumanML3D/new_joint_vecs/004822.npy')
# print(raw_motion[0,:5])

# humanml_mean = np.load('dataset/HumanML3D/Mean.npy')[None, ...] # (1,1,263) dataset/HumanML3D/Mean.npy
# humanml_std = np.load('dataset/HumanML3D/Std.npy')[None, ...]  # (1,1,263)
# meta_mean = np.load('checkpoints/t2m/Comp_v6_KLD005/meta/mean.npy')[None, ...]
# meta_std = np.load('checkpoints/t2m/Comp_v6_KLD005/meta/std.npy')[None, ...]
# norm_motion1 = (raw_motion - humanml_mean) / humanml_std
# norm_motion2 = (raw_motion - meta_mean) / meta_std
# print(norm_motion1[0,:5])
# print(norm_motion2[0,:5])

mean = np.load('checkpoints/t2m/Comp_v6_KLD005/meta/mean.npy')
Mean = np.load('dataset/HumanML3D/Mean.npy')
meta_std = np.load('checkpoints/t2m/Comp_v6_KLD005/meta/std.npy')
Std = np.load('dataset/HumanML3D/Std.npy')
std = Std.copy()
joints_num = 22
std[0:1] = Std[0:1] / 5
# root_linear_velocity (B, seq_len, 2)
std[1:3] = Std[1:3] / 5
# root_y (B, seq_len, 1)
std[3:4] = Std[3:4] / 5
# ric_data (B, seq_len, (joint_num - 1)*3)
std[4: 4 + (joints_num - 1) * 3] = Std[4: 4 + (joints_num - 1) * 3] / 1.0
# rot_data (B, seq_len, (joint_num - 1)*6)
std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = Std[4 + (joints_num - 1) * 3: 4 + (
            joints_num - 1) * 9] / 1.0
# local_velocity (B, seq_len, joint_num*3)
std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = Std[
                                                                            4 + (joints_num - 1) * 9: 4 + (
                                                                                        joints_num - 1) * 9 + joints_num * 3] / 1.0
# foot contact (B, seq_len, 4)
std[4 + (joints_num - 1) * 9 + joints_num * 3:] = Std[
                                                    4 + (joints_num - 1) * 9 + joints_num * 3:] / 5
x = mean - Mean
y = std - Std
a = 1
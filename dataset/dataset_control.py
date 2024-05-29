import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm

from torch.utils.data._utils.collate import default_collate
from utils.word_vectorizer import WordVectorizer
from options.get_eval_option import get_opt
from utils.motion_process import recover_root_rot_pos, recover_from_ric
from utils.metrics import cross_combination_joints
import sys

def collate_fn(batch):
    if batch[0][-1] is None:
        batch = [b[:-1] for b in batch]
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text motion matching model, and evaluations'''
class ControlDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer, mode, 
                 control_joint=0, density=100, dataset_name='t2m', normalize_traj=False, 
                 multi_joint_control=False, unit_length=None, codebook_dir=None, codebook_size=512, dense_control=False):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.split_file = split_file
        self.mode = mode
        self.control_joint = control_joint
        self.density = density
        self.normalize_traj = normalize_traj
        self.multi_joint_control = multi_joint_control
        print('self.control_joint = ', self.control_joint)
        print('self.density = ', self.density)
        assert  0 <= self.density <= 100, "density should be in [0, 100], got {}".format(self.density)

        # 以下是VQ相关的
        self.unit_length = unit_length # 1个token对应实际的动作长度，如果VQVAE里下采样2次，那么unit_length=4
        self.codebook_dir = codebook_dir # 训练前预存的motion tokens的路径 output/vq/vq_name/codebook
        self.codebook_size = codebook_size
        self.dense_control = dense_control
        self.motion_end_idx = codebook_size
        self.motion_pad_idx = codebook_size + 1
        self.max_token_length = 26 if unit_length == 8 else 50
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24
        

        fps = 20 if self.opt.dataset_name == 't2m' else 12.5 # HumanML3D帧率20; KIT帧率12.5

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        # id_list = id_list[:111]  if sys.gettrace() else id_list
        # if self.mode == 'debug':
        # id_list = id_list[:32]
        
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if self.codebook_dir:
                    motion_token =  np.load(pjoin(self.codebook_dir, name + '.npy'))[0] # 因为读进来是(1,L) 实际上只有1个
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                # 这部分的意思是，一段动作，但是有的文本对应的动作段是有起始点和结束点的
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag  # 起始秒 from_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag # 结束秒 to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * fps) : int(to_tag * fps)] # 起始秒和结束秒 乘上帧率 KIT帧率是12.5
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200): # 过滤长度过短过长的动作
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                                if self.codebook_dir:
                                    motion_token_ = motion_token[int(f_tag * fps / unit_length) : int(to_tag * fps / unit_length)]
                                    data_dict[new_name]['motion_token'] = motion_token_

                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    if self.codebook_dir:
                        data_dict[name]['motion_token'] = motion_token
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        # name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1])) # 把名字长度二元组依长度升序排列
        name_list = new_name_list # debug用
        # length_list = length_list # debug用

        self.mean = mean
        self.std = std
        if 'HumanML3D' in opt.data_root:
            spatial_norm_path = './dataset/humanml_spatial_norm'
        elif 'KIT' in opt.data_root:
            spatial_norm_path = './dataset/kit_spatial_norm'
        else:
            raise NotImplementedError('unknown dataset')
        # 全局xyz的均值和方差；原本HumanML3D的Mean.npy是相对xyz的均值和方差
        self.raw_mean = np.load(pjoin(spatial_norm_path, 'Mean_raw.npy')).reshape(22,3)
        self.raw_std = np.load(pjoin(spatial_norm_path, 'Std_raw.npy')).reshape(22,3)
        
        self.data_dict = data_dict
        self.name_list = name_list
        print(f'=== total {len(self.data_dict)} data')
        

    
    def random_mask(self, joints, n_joints=22):
        if n_joints == 22:
            # humanml3d
            controllable_joints = np.array([0, 10, 11, 15, 20, 21])
            joints_name = np.array(['pelvis', 'left_foot', 'right_foot', 'head', 'left_wrist', 'right_wrist'])
        else:
            # kit
            {1:'root', 2:'BP', 3:'BT', 4:'BLN', 5:'BUN', 6:'LS', 7:'LE', 8:'LW', 9:'RS', 10:'RE', 11:'RW', 12:'LH', 13:'LK', 14:'LA', 15:'LMrot', 16:'LF', 17:'RH', 18:'RK', 19:'RA', 20:'RMrot', 21:'RF'}
            choose_one = ['root', 'BUN', 'LW', 'RW', 'LF', 'RF']
            controllable_joints = np.array([0, 4, 7, 10, 15, 20])

        # 选择控制关节
        if isinstance(self.control_joint, list):
            choose_joint = self.control_joint
        else:
            if self.control_joint >= 0: # 默认 -1, 表示随机选取
                choose_joint = [self.control_joint]
            else:
                num_joints = len(controllable_joints)
                if self.multi_joint_control:
                    num_joints_control = np.random.choice(np.arange(1, num_joints+1), 1) # 1~6  多关节控制
                else:
                    num_joints_control = 1
                choose_joint = np.random.choice(num_joints, num_joints_control, replace=False) # 选择控制的关节点
                choose_name = joints_name[choose_joint]
                choose_joint = controllable_joints[choose_joint]

        # 选择控制帧比例
        length = joints.shape[0]
        choose_seq_num = np.random.choice(length - 1, 1) + 1
        if self.density:
            if self.density in [1, 2, 5]:
                choose_seq_num = self.density
            else:
                choose_seq_num = int(length * self.density / 100)
        choose_seq = np.random.choice(length, choose_seq_num, replace=False)
        choose_seq.sort()
        
        # 生成控制mask
        mask_seq = np.zeros((self.max_motion_length, n_joints, 3)).astype(bool) # (196,22,3)
        mask_seq_263 = np.zeros((self.max_motion_length, 263)).astype(bool) # (196,263)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True
            mask_seq_263[choose_seq, :4] = True # root
            mask_seq_263[choose_seq, 4+3*(cj-1):4+3*cj] = True # ric  21*3

        # normalize
        if self.normalize_traj: # omnicontrol最原本就是不归一化轨迹
            joints = (joints - self.raw_mean) / self.raw_std
        joints = joints * mask_seq[:length, ...]
        return joints, mask_seq_263, mask_seq

    def random_mask_train(self, joints, n_joints=22):
        if n_joints == 22:
            controllable_joints = np.array([0, 10, 11, 15, 20, 21]) # 盆骨，左脚，右脚，头，左手腕，右手腕
            joints_name = np.array(['pelvis', 'left_foot', 'right_foot', 'head', 'left_wrist', 'right_wrist'])
        else:
            {1:'root', 2:'BP', 3:'BT', 4:'BLN', 5:'BUN', 6:'LS', 7:'LE', 8:'LW', 9:'RS', 10:'RE', 11:'RW', 12:'LH', 13:'LK', 14:'LA', 15:'LMrot', 16:'LF', 17:'RH', 18:'RK', 19:'RA', 20:'RMrot', 21:'RF'}
            choose_one = ['root', 'BUN', 'LW', 'RW', 'LF', 'RF']
            controllable_joints = np.array([0, 4, 7, 10, 15, 20])

        # 选择控制关节
        num_joints = len(controllable_joints)
        if self.multi_joint_control:
            num_joints_control = np.random.choice(np.arange(1, num_joints+1), 1) # 1~6  多关节控制
        else:
            num_joints_control = 1

        choose_joint = np.random.choice(num_joints, num_joints_control, replace=False) # 选择控制的关节点
        choose_name = joints_name[choose_joint]
        choose_joint = controllable_joints[choose_joint]

        # print('control joints = ', choose_joint, choose_name)

        # 选择控制帧比例
        length = joints.shape[0]
        choose_seq_num = np.random.choice(length - 1, 1) + 1 # 随机设定控制的帧数 范围 [1,L-1]
        if self.dense_control and self.mode != 'train': # 全部帧控制
            choose_seq_num = length
        choose_seq = np.random.choice(length, choose_seq_num, replace=False) # 根据帧数选择控制的时刻帧
        choose_seq.sort()
        # print('frames percent = ', choose_seq_num/length)

        # 生成控制mask
        mask_seq = np.zeros((self.max_motion_length, n_joints, 3)).astype(bool) # (196,22,3)
        mask_seq_263 = np.zeros((self.max_motion_length, 263)).astype(bool) # (196,263)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True
            mask_seq_263[choose_seq, :4] = True # root
            mask_seq_263[choose_seq, 4+3*(cj-1):4+3*cj] = True # ric  21*3
            # if cj!=0:
            #     mask_seq_263[choose_seq, 67+6*(cj-1):67+6*cj] = True # rot     21*6
            # mask_seq_263[choose_seq, 193+3*cj:193+3*(cj+1)] = True # vel   22*3

        # normalize
        if self.normalize_traj: # omnicontrol最原本就是不归一化轨迹
            joints = (joints - self.raw_mean) / self.raw_std
        joints = joints * mask_seq[:length, ...]
        return joints, mask_seq_263, mask_seq
    


    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        # idx = 29
        data = self.data_dict[self.name_list[idx]]
        # data = self.data_dict['000007'] 
        
        # print('idx = ', idx, self.name_list[idx])
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # 句子短，补SOS和EOS token，然后补unknown token至固定长度
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # 句子场，固定切割到固定长度，再补SOS EOS
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)        # (22,15)   22是预设的句子最大长度20加上2个SOS和EOS
        word_embeddings = np.concatenate(word_embeddings, axis=0)  # (22,300)

        # 将动作长度截取为unit_length即4的整数倍，并通过coin2引入一些随机，无需细抠
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'
        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        n_joints = 22 if motion.shape[-1] == 263 else 21
        # hint is global position of the controllable joints
        joints = recover_from_ric(torch.from_numpy(motion).float(), n_joints) # (L, 22, 3)  每个关节点的全局坐标
        joints = joints.numpy()
        

        # control any joints at any time
        if self.mode == 'train' or self.mode == 'debug':
            # hint = self.random_mask_train_cross(joints, n_joints)
            hint, traj_mask_263, traj_mask = self.random_mask_train(joints, n_joints)
        else:
            hint, traj_mask_263, traj_mask  = self.random_mask(joints, n_joints) # joints: (L,22,3)

        hint = hint.reshape(hint.shape[0], -1) # (L,22*3)

        # motion 263的归一化
        motion = (motion - self.mean) / self.std
        
        if m_length < self.max_motion_length: # 固定输出动作长度为max_length ！！
            hint   = np.concatenate([hint, np.zeros((self.max_motion_length - m_length, hint.shape[1])) ], axis=0)
            motion = np.concatenate([motion, np.zeros((self.max_motion_length - m_length, motion.shape[1])) ], axis=0)

        hint = hint.astype(np.float32).reshape(self.max_motion_length, n_joints, 3)
        motion = motion.astype(np.float32)
        
        # traj_mask_263 = self.get_choose_joint_mask2(traj_mask)

        # 确保取得的轨迹以及traj_mask正确
        if self.normalize_traj:
            # joints: L,22,3
            # hint: 196,22,3
            assert np.allclose(joints * traj_mask[:m_length, ...] , ((hint * self.raw_std + self.raw_mean) * traj_mask)[:m_length, ...], atol=1e-6)
        else:
            assert (joints*traj_mask[:m_length, ...] - hint[:m_length, ...]).sum() == 0

        if not self.codebook_dir: # 如果是训练2阶段的AE，即无codebook_dir，这里就返回
            return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), hint, traj_mask_263, traj_mask
        
        ### 后面都是针对motion_token的处理
        motion_token = data['motion_token']
        if np.random.choice([False, False, True]):
            # drop one token at the head or tail
            if np.random.choice([True, False]):
                motion_token = motion_token[:-1]
            else:
                motion_token = motion_token[1:]
        motion_token_len = motion_token.shape[0]
        if motion_token_len<10:
            a = 1
        if motion_token_len==0:
            b = 1
        if motion_token_len+1 < self.max_token_length:  # unit_length=4时，最大长度内，有效token和pad最多49个，1个end
            motion_token = np.concatenate([motion_token, np.ones((1), dtype=int) * self.motion_end_idx, 
                                           np.ones((self.max_token_length-1-motion_token_len), dtype=int) * self.motion_pad_idx], axis=0)
        else:
            motion_token = np.concatenate([motion_token, np.ones((1), dtype=int) * self.motion_end_idx], axis=0)
        return caption, motion, motion_token, motion_token_len, hint, traj_mask

# A wrapper class for t2m original dataset for MDM purposes
class HumanML3DControl(data.Dataset):
    def __init__(self, mode, codebook_dir, datapath='./dataset/humanml_opt.txt', split="train", control_joint=0, normalize_traj=False, multi_joint_control=False, unit_length=None, codebook_size=512, density=100, dense_control=False, **kwargs):
        
        self.split = split
        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = f'.'
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = None  # torch.device('cuda:4') # This param is not in use in this context
        opt = get_opt(dataset_opt_path, device)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        # opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        # opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        # opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = './dataset'
        self.opt = opt
        self.dataset_name = opt.dataset_name
        print('Loading dataset %s ...' % opt.dataset_name)

        if mode == 'gt':
            # used by T2M models (including evaluators)
            self.mean = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))
        elif mode in ['train', 'eval', 'text_only']:
            self.mean = np.load(pjoin(opt.data_root, 'Mean.npy')) # dataset/HumanML3D/Mean.npy
            self.std = np.load(pjoin(opt.data_root, 'Std.npy'))

        if mode == 'eval':
            # used by T2M models (including evaluators)
            # this is to translate their norms to ours
            self.mean_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))

        self.split_file = pjoin(opt.data_root, f'{split}.txt') # dataset/HumanML3D/train.txt

        self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
        self.t2m_dataset = ControlDataset(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer, mode,  
                                          control_joint=control_joint, density=density, dataset_name=self.dataset_name, 
                                          normalize_traj=normalize_traj, multi_joint_control=multi_joint_control,
                                          unit_length=unit_length, codebook_dir=codebook_dir, codebook_size=codebook_size, dense_control=dense_control)
        self.num_actions = 1 # dummy placeholder


    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()

# A wrapper class for t2m original dataset for MDM purposes
class KIT(HumanML3DControl):
    def __init__(self, mode, datapath='./dataset/kit_opt.txt', split="train", **kwargs):
        super(KIT, self).__init__(mode, datapath, split, **kwargs)

def DataLoader(batch_size, args, shuffle=False, codebook_dir=None, mode='train', split='train', num_workers=8, drop_last=True) : 
    
    dataset = HumanML3DControl(mode, codebook_dir, split=split, control_joint=args.control_joint, 
                               normalize_traj=args.normalize_traj, density=args.density, 
                               multi_joint_control=args.multi_joint_control, unit_length=2*args.down_t, 
                               codebook_size=args.nb_code, dense_control=args.dense_control)
    if batch_size == 1:
        num_workers = 0
    train_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, drop_last = drop_last)
    return train_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

if __name__ == '__main__':
    train_loader = DataLoader(batch_size=1, mode='train')
    train_loader_iter = cycle(train_loader)
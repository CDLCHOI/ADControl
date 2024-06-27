import torch
from data_loaders.humanml.networks.modules import *
from torch.utils.data import Dataset
import clip
from utils.mask_utils import generate_src_mask
import torch.nn.functional as F
# from exit.utils import visualize_2motions
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper

class CompADCGeneratedDataset(Dataset):
    def __init__(self, args, gen_loader, clip_model, diffusion_root, diffusion, mm_num_samples, mm_num_repeats, num_samples_limit):
        self.eval_wrapper = EvaluatorMDMWrapper('humanml',torch.device('cuda'))
        self.args = args
        self.gen_loader = gen_loader
        self.dataset = gen_loader.dataset
        assert mm_num_samples < len(gen_loader.dataset)
        real_num_batches = len(gen_loader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // gen_loader.batch_size + 1
        print('real_num_batches', real_num_batches)
        # 读取数据集的均值和方差
        # mean = np.load('dataset/HumanML3D/Mean.npy')[None, ...] 
        # std = np.load('dataset/HumanML3D/Std.npy')[None, ...]
        # humanml_mean = torch.from_numpy(mean)[None, ...].cuda()
        # humanml_std = torch.from_numpy(std)[None, ...].cuda()
        # raw_mean = torch.from_numpy(np.load('dataset/humanml_spatial_norm/Mean_raw.npy')).cuda()[None, None, ...].view(1,1,22,3) 
        # raw_std = torch.from_numpy(np.load('dataset/humanml_spatial_norm/Std_raw.npy')).cuda()[None, None, ...].view(1,1,22,3)
        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // gen_loader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs = ', mm_idxs)
        # samples = []
        # gt_motions = []
        # real_lengths = []
        for i, batch in enumerate(self.gen_loader):
            print(f'{i}/{len(self.gen_loader)}')
            word_embeddings, pos_one_hots, clip_text, sent_len, gt_motion, real_length, txt_tokens, traj, traj_mask_263, traj_mask = batch
            txt_tokens = [t.split('_') for t in txt_tokens]
            b, max_length, num_features = gt_motion.shape
            gt_motion = gt_motion.cuda()
            real_length = real_length.cuda()
            # real_lengths.append(real_length)
            traj = traj.cuda()
            traj_mask = traj_mask.cuda()
            traj_mask_263 = traj_mask_263.cuda()
            real_mask = generate_src_mask(max_length, real_length) # (b,196)
            gt_ric = gt_motion[..., :67]
            
            #encode text
            text = clip.tokenize(clip_text, truncate=True).cuda()        
            text_emb, word_emb = clip_model(text) # (b,512) 
            
            model_kwargs = {}
            model_kwargs['traj'] = traj
            model_kwargs['text_emb'] = text_emb
            model_kwargs['word_emb'] = word_emb
            model_kwargs['traj_mask'] = traj_mask
            model_kwargs['traj_mask_263'] = traj_mask_263
            model_kwargs['gt_motion'] = gt_motion
            model_kwargs['traj'] = traj
            model_kwargs['real_mask'] = real_mask
            model_kwargs['clip_text'] = clip_text

            is_mm = i in mm_idxs
            repeat_times = mm_num_repeats if is_mm else 1
            mm_motions = []

            # stage1
            if args.roottype == 'omni67': 
                pred_ric = diffusion_root.p_sample_loop(partial_emb=None, model_kwargs=model_kwargs,batch_size=args.batch_size)

            # control_id = traj_mask[0].sum(0).sum(-1).nonzero()
            # if args.normalize_traj:
            #     traj = traj * raw_std + raw_mean

            # stage2
            for t in range(repeat_times):
                partial_emb = torch.zeros_like(gt_motion, device=gt_motion.device)
                partial_emb[..., :67] = pred_ric  
                # partial_emb[..., :67] = gt_ric ##debug 

                if args.modeltype == 'semboost':
                    sample = diffusion.p_sample_loop(partial_emb, with_control=True, model_kwargs=model_kwargs, batch_size=args.batch_size) # (b, 196, 263)
                    # sample = gt_motion

                if t == 0:
                    sub_dicts = [{'motion': sample[bs_i].squeeze().cpu().numpy(),
                                'length': real_length[bs_i].cpu().numpy(),
                                'caption': clip_text[bs_i],
                                'hint': '',
                                'tokens': txt_tokens[bs_i],
                                # 'cap_len': len(txt_tokens[bs_i]), # 这里是错的，len(txt_tokens[bs_i])=22是data_control里已经填充到固定长度了
                                'cap_len': sent_len[bs_i].item(),
                                } for bs_i in range(gen_loader.batch_size)]
                    generated_motion += sub_dicts

                if is_mm:
                    mm_motions += [{'motion': sample[bs_i].squeeze().cpu().numpy(),
                                    'length': real_length[bs_i].cpu().numpy(),
                                    } for bs_i in range(gen_loader.batch_size)]
            
            if is_mm:
                mm_generated_motions += [{
                                'caption': clip_text[bs_i],
                                'tokens': txt_tokens[bs_i],
                                # 'cap_len': len(txt_tokens[bs_i]),
                                'cap_len': sent_len[bs_i].item(),
                                'mm_motions': mm_motions[bs_i::gen_loader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
                                } for bs_i in range(gen_loader.batch_size)]
        
        
        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = gen_loader.dataset.w_vectorizer

    def __len__(self):
        return len(self.generated_motion)
    
    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens, hint = data['motion'], data['length'], data['caption'], data['tokens'], data['hint'],
        sent_len = data['cap_len']

        if self.dataset.mode == 'eval':
            normed_motion = motion
            denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
            renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
            motion = renormed_motion
            # This step is needed because T2M evaluators expect their norm convention

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), hint
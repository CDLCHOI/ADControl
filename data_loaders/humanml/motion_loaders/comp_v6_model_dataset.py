import torch
from data_loaders.humanml.networks.modules import *
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import dist_util
from utils.motion_process import recover_from_ric
import clip
from utils.mask_utils import generate_src_mask, visualize_2motions, vis_motion
import torch.nn.functional as F
# from exit.utils import visualize_2motions
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper

class CompADCGeneratedDataset(Dataset):
    def __init__(self, args, gt_loader, clip_model, diffusion_root, diffusion, mm_num_samples, mm_num_repeats, num_samples_limit):
        self.eval_wrapper = EvaluatorMDMWrapper('humanml',torch.device('cuda'))
        self.args = args
        self.gt_loader = gt_loader
        self.dataset = gt_loader.dataset
        assert mm_num_samples < len(gt_loader.dataset)
        real_num_batches = len(gt_loader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // gt_loader.batch_size + 1
        print('real_num_batches', real_num_batches)
        # 读取数据集的均值和方差
        mean = np.load('dataset/HumanML3D/Mean.npy')[None, ...] 
        std = np.load('dataset/HumanML3D/Std.npy')[None, ...]
        humanml_mean = torch.from_numpy(mean)[None, ...].cuda()
        humanml_std = torch.from_numpy(std)[None, ...].cuda()
        raw_mean = torch.from_numpy(np.load('dataset/humanml_spatial_norm/Mean_raw.npy')).cuda()[None, None, ...].view(1,1,22,3) 
        raw_std = torch.from_numpy(np.load('dataset/humanml_spatial_norm/Std_raw.npy')).cuda()[None, None, ...].view(1,1,22,3)
        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // gt_loader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs = ', mm_idxs)
        # samples = []
        # gt_motions = []
        # real_lengths = []
        for i, batch in enumerate(self.gt_loader):
            print(f'{i}/{len(self.gt_loader)}')
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

            #debug
            # t, weights = diffusion.schedule_sampler.sample(b, gt_motion.device) # timestep
            # t = torch.tensor([999]*b).cuda()
            # x0 = gt_motion
            # noise = torch.randn_like(x0)
            # xt = diffusion.q_sample(x0, t, noise=noise)
            # masked_xt = xt
            # pred_x0 = diffusion.model(masked_xt, t, text_emb, word_emb) # (b,196,263)
            
            # fid = evaluate_fid_in_train(self.eval_wrapper, gt_motion, real_length, pred_x0, '/home/shenbo/projects/OmniControl/log.txt')
            
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

            # # 采样根节点轨迹
            # if args.roottype == 'omni67': 
            #     pred_ric = diffusion_root.p_sample_loop(partial_emb=None, model_kwargs=model_kwargs,batch_size=args.batch_size)

            # control_id = traj_mask[0].sum(0).sum(-1).nonzero()
            if args.normalize_traj:
                traj = traj * raw_std + raw_mean

            for t in range(repeat_times):
                # 采样动作
                partial_emb = torch.zeros_like(gt_motion, device=gt_motion.device)
                # partial_emb[..., :67] = pred_ric  
                partial_emb[..., :67] = gt_ric ##debug 

                if 'diffmae_stage2' in args.modeltype or args.modeltype == 'diffmdm' or args.modeltype == 'semboost':
                    sample = diffusion.p_sample_loop(partial_emb, with_control=True, model_kwargs=model_kwargs, batch_size=args.batch_size) # (b, 196, 263)
                    # sample = gt_motion

                if t == 0:
                    sub_dicts = [{'motion': sample[bs_i].squeeze().cpu().numpy(),
                                'length': real_length[bs_i].cpu().numpy(),
                                'caption': clip_text[bs_i],
                                'hint': '',
                                'tokens': txt_tokens[bs_i],
                                'cap_len': len(txt_tokens[bs_i]),
                                } for bs_i in range(gt_loader.batch_size)]
                    generated_motion += sub_dicts

                if is_mm:
                    mm_motions += [{'motion': sample[bs_i].squeeze().cpu().numpy(),
                                    'length': real_length[bs_i].cpu().numpy(),
                                    } for bs_i in range(gt_loader.batch_size)]
            
            if is_mm:
                mm_generated_motions += [{
                                'caption': clip_text[bs_i],
                                'tokens': txt_tokens[bs_i],
                                'cap_len': len(txt_tokens[bs_i]),
                                'mm_motions': mm_motions[bs_i::gt_loader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
                                } for bs_i in range(gt_loader.batch_size)]
        
        
        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = gt_loader.dataset.w_vectorizer

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

class CompMDMGeneratedDataset(Dataset):

    def __init__(self, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1.):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        assert mm_num_samples < len(dataloader.dataset)
        use_ddim = False  # FIXME - hardcoded
        clip_denoised = False  # FIXME - hardcoded
        self.max_motion_length = max_motion_length
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )

        real_num_batches = len(dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // dataloader.batch_size + 1
        print('real_num_batches', real_num_batches)

        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs', mm_idxs)

        model.eval()


        with torch.no_grad():
            for i, (motion, model_kwargs) in tqdm(enumerate(dataloader)): # 这里加载的就是gt数据，model_kwargs即一系列condition
                for k, v in model_kwargs['y'].items():
                    if torch.is_tensor(v):
                        model_kwargs['y'][k] = v.to(dist_util.dev())

                if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                    break

                tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

                # add CFG scale to batch
                if scale != 1.:
                    model_kwargs['y']['scale'] = torch.ones(motion.shape[0],
                                                            device=dist_util.dev()) * scale

                mm_num_now = len(mm_generated_motions) // dataloader.batch_size
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):

                    sample = sample_fn(
                        model,
                        motion.shape,
                        clip_denoised=clip_denoised,
                        model_kwargs=model_kwargs,
                        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                        init_image=None,
                        progress=False,
                        dump_steps=None,
                        noise=None,
                        const_noise=False,
                        # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
                    )

                    if t == 0:
                        sub_dicts = [{'motion': sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
                                    'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'hint': model_kwargs['y']['hint'][bs_i].cpu().numpy() if 'hint' in model_kwargs['y'] else None,
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    } for bs_i in range(dataloader.batch_size)]
                        generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [{'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                                        'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                        } for bs_i in range(dataloader.batch_size)]

                if is_mm:
                    mm_generated_motions += [{
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    'mm_motions': mm_motions[bs_i::dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
                                    } for bs_i in range(dataloader.batch_size)]


        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = dataloader.dataset.w_vectorizer


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
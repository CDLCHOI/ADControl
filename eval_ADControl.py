import sys
sys.path.append('.')
import options.option_transformer as option_trans
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
os.environ['OMP_NUM_THREADS'] = '8'
from utils import dist_util
from utils.fixseed import fixseed
import torch
from data_loaders.humanml.utils.metrics import *
from datetime import datetime
import numpy as np
from collections import OrderedDict
from data_loaders.humanml.motion_loaders.model_motion_loaders import get_control_dataset
import clip
import time
from dataset import dataset_control
import warnings
warnings.filterwarnings('ignore')
# from diffusion import logger
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from utils.mask_utils import calc_grad_scale, TextCLIP, calc_loss_xyz, load_ckpt
from utils.model_util import create_gaussian_diffusion_simple, get_logger
from models.cfg_sampler import ClassifierFreeSampleModelADC,ClassifierFreeSampleModel

def refine(args):
    args.roottype = 'root'
    args.exp_name = 'debug_model'
    args.print_iter = 1
    args.overwrite = True
    args.total_iter_ROOT = 20
    args.total_iter_ED = 1 
    args.dense_control = False
    return args

def evaluate_matching_score(eval_wrapper, motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    print('========== Evaluating Matching Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                if len(batch) == 7:
                    word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch
                elif motion_loader_name == 'ground truth':
                    word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _, _, _, _= batch
                else:
                    word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _, _ = batch
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens
                )
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                     motion_embeddings.cpu().numpy())
                matching_score_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}')
        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid(eval_wrapper, groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _, _, _, _= batch
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions,
                m_lens=m_lens
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    # print(gt_mu)
    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        # print(mu)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file, diversity_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict


def evaluate_multimodality(eval_wrapper, mm_motion_loaders, file, mm_num_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                motions, m_lens = batch
                motion_embedings = eval_wrapper.get_motion_embeddings(motions[0], m_lens[0])
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval



def evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times, diversity_times, mm_num_times, run_mm=False):
    with open(log_file, 'a+') as f:
        all_metrics = OrderedDict({'Matching Score': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Diversity': OrderedDict({}),
                                   'MultiModality': OrderedDict({}),
                                   'Control_l2': OrderedDict({}),
                                   'Skating Ratio': OrderedDict({}),
                                   'Trajectory Error': OrderedDict({})})

        for replication in range(replication_times):
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader, mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(eval_wrapper, motion_loaders, f)

            # print(f'Time: {datetime.now()}')
            # print(f'Time: {datetime.now()}', file=f, flush=True)
            # control_l2_dict, skating_ratio_dict, trajectory_score_dict = evaluate_control(motion_loaders, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(eval_wrapper, gt_loader, acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f, diversity_times)

            if run_mm:
                print(f'Time: {datetime.now()}')
                print(f'Time: {datetime.now()}', file=f, flush=True)
                mm_score_dict = evaluate_multimodality(eval_wrapper, mm_motion_loaders, f, mm_num_times)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

        #     for key, item in mat_score_dict.items():
        #         if key not in all_metrics['Matching Score']:
        #             all_metrics['Matching Score'][key] = [item]
        #         else:
        #             all_metrics['Matching Score'][key] += [item]

        #     for key, item in R_precision_dict.items():
        #         if key not in all_metrics['R_precision']:
        #             all_metrics['R_precision'][key] = [item]
        #         else:
        #             all_metrics['R_precision'][key] += [item]

        #     for key, item in fid_score_dict.items():
        #         if key not in all_metrics['FID']:
        #             all_metrics['FID'][key] = [item]
        #         else:
        #             all_metrics['FID'][key] += [item]

        #     for key, item in div_score_dict.items():
        #         if key not in all_metrics['Diversity']:
        #             all_metrics['Diversity'][key] = [item]
        #         else:
        #             all_metrics['Diversity'][key] += [item]
        #     if run_mm:
        #         for key, item in mm_score_dict.items():
        #             if key not in all_metrics['MultiModality']:
        #                 all_metrics['MultiModality'][key] = [item]
        #             else:
        #                 all_metrics['MultiModality'][key] += [item]


        # # print(all_metrics['Diversity'])
        # mean_dict = {}
        # for metric_name, metric_dict in all_metrics.items():
        #     print('========== %s Summary ==========' % metric_name)
        #     print('========== %s Summary ==========' % metric_name, file=f, flush=True)
        #     for model_name, values in metric_dict.items():
        #         # print(metric_name, model_name)
        #         mean, conf_interval = get_metric_statistics(np.array(values), replication_times)
        #         mean_dict[metric_name + '_' + model_name] = mean
        #         # print(mean, mean.dtype)
        #         if isinstance(mean, np.float64) or isinstance(mean, np.float32):
        #             print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
        #             print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
        #         elif metric_name == 'Trajectory Error':
        #             traj_err_key = ["traj_fail_20cm", "traj_fail_50cm", "kps_fail_20cm", "kps_fail_50cm", "kps_mean_err(m)"]
        #             line = f'---> [{model_name}]'
        #             for i in range(len(mean)): # zip(traj_err_key, mean):
        #                 line += '(%s): Mean: %.4f CInt: %.4f; ' % (traj_err_key[i], mean[i], conf_interval[i])
        #             print(line)
        #             print(line, file=f, flush=True)
        #         elif isinstance(mean, np.ndarray):
        #             line = f'---> [{model_name}]'
        #             for i in range(len(mean)):
        #                 line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
        #             print(line)
        #             print(line, file=f, flush=True)
        # return mean_dict



if __name__ == '__main__':
    args = option_trans.get_args_parser()
    fixseed(args.seed)
    args = refine(args)
    # args.device = 0
    args.guidance_param = 2.5
    args.batch_size = 32 # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!

    # 需要修改的地方
    args.dataset = 'humanml' # choices=['humanml', 'kit', 'humanact12', 'uestc'], type=str,
    args.control_joint = 0
    args.density = 100
    args.eval_mode = 'ADControl'
    # args.modeltype = 'diffmdm'
    args.modeltype = 'semboost'
    if args.modeltype == 'diffmae_stage2_2':
        args.resume_trans = 'output/0509_diffmae_stage2_2_E8D0_multicontrol_pretrain/net_last.pth'
        args.resume_trans = '/home/shenbo/projects/OmniControl/output/0518_diffmae_stage2_2_L2loss/net_last.pth'
    elif args.modeltype == 'diffmdm':
        # args.resume_trans = '/home/shenbo/projects/OmniControl/output/0519_diffmdm/net_last.pth'
        args.resume_trans = '/home/shenbo/projects/OmniControl/savemdm/model000475000.pt'
    elif args.modeltype == 'semboost':
        # args.resume_trans = '/home/shenbo/projects/OmniControl/output/0520_semboost/net_last.pth'
        args.resume_trans = '/home/shenbo/projects/OmniControl/output/0520_semboost_latest/net_last.pth'
        # args.resume_trans = '/home/shenbo/projects/OmniControl/output/0520_semboost_noxyzloss/net_last.pth'
    # args.resume_trans = '/home/shenbo/projects/OmniControl/output/stage2_mdm/net_last.pth'
    # args.resume_root = 'output/0514_omni67_normtraj_multicontrol/net_last.pth'
    args.resume_root = './output/0518_omni67_multi_partxyz/net_last.pth'; args.roottype = 'omni67'
    args.normalize_traj=True # 归一化轨迹再输入
    if args.eval_mode == 'ADControl':
        num_samples_limit = 1000
        run_mm =False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 300
        replication_times = 1 # 重复测试次数
    else:
        raise ValueError()

    name = os.path.basename(os.path.dirname(args.resume_trans))
    niter = os.path.basename(args.resume_trans).replace('model', '').replace('.pt', '')
    log_file = f'output/log/evalADControl_gtric_joint_{args.control_joint}_density_{args.density}.log'
    log_file = f'output/log/evalADControl_joint_{args.control_joint}_density_{args.density}.log'
    log_file = f'output/log/evalADControl_gtric_with_renorm.log'
    if sys.gettrace():
        log_file = f'output/log/1.log'
    logger = get_logger('', file_path=log_file)
    logger.info(f'log_file = {log_file}')
    logger.info(f'args.resume_root = {args.resume_root}')
    logger.info(f'args.resume_trans = {args.resume_trans}')
    logger.info(f'control joint = {args.control_joint}, density = {args.density}')
    logger.info(f'无1阶段，使用gtric，有renorm') # 对文件的说明

    print(f'Eval mode [{args.eval_mode}]')
    


    ##### ---- CLIP ---- #####
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
    clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_model = TextCLIP(clip_model)

    # 根节点网络
    if args.roottype == 'omni67':
        from models.omni67 import CMDM
        net_root = CMDM(args, args.roottype)
    diffusion_root = create_gaussian_diffusion_simple(args, net_root, args.roottype, clip_model)

    load_ckpt(net_root, args.resume_root, key='trans')
    net_root.cuda()
    net_root.eval()

    # 2阶段网络
    if args.modeltype == 'diffmae_stage2_2':
        from models.diffmae_2 import DiffMAE2
        net = DiffMAE2(dataset=args.dataset_name, args=args, num_layers_E=8, num_layers_D=0)
    elif args.modeltype == 'semboost':
        from models.semanticboost import SemanticBoost
        from utils.model_util import get_semanticboost_args
        net = SemanticBoost(**get_semanticboost_args(args))

    load_ckpt(net, args.resume_trans, key='trans')

    if args.guidance_param != 1:
        if args.modeltype == 'diffmae_stage2_2':
            net = ClassifierFreeSampleModelADC(net, args=args)
        elif args.modeltype == 'diffmdm' or args.modeltype == 'semboost':
            net = ClassifierFreeSampleModel(net)
            # net = ClassifierFreeSampleModelADC(net, args=args)
            
    diffusion = create_gaussian_diffusion_simple(args, net, args.modeltype, clip_model)
    net.cuda()
    net.eval()
    
    #评估生成数据集部分  shuffle = False
    gt_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode='gt', split='test', shuffle=False, num_workers=0, drop_last=True)
    gen_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode='eval', split='test', shuffle=False, num_workers=0, drop_last=True)
    eval_motion_loaders = {
        ## HumanML3D Dataset##
        'vald': lambda: get_control_dataset(
            args, gen_loader, clip_model, diffusion_root, diffusion, mm_num_samples, mm_num_repeats, num_samples_limit
        )
    }
    eval_wrapper = EvaluatorMDMWrapper(args.dataset, torch.device('cuda'))
    evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times, diversity_times, mm_num_times, run_mm=run_mm)

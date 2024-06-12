import os 
import sys
import options.option_transformer as option_trans
args = option_trans.get_args_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)  # 设定GPU
import torch
import torch.nn as nn
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
import clip
import json
from utils.model_util import initial_optim, get_logger
from utils.mask_utils import load_ckpt
from dataset import dataset_control
import warnings
warnings.filterwarnings('ignore')
import shutil

if __name__ == '__main__':
    # 训练前准备
    args.out_dir = pjoin(args.out_dir, args.exp_name) # output/trans_exp_name
    if args.overwrite and os.path.exists(args.out_dir):
        assert not os.path.exists(pjoin(args.out_dir, 'net_last.pth')), f'net_last.pth exist in {args.out_dir}'
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok = True)

    # logger
    logger = get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True)) # args所有输出到log
    logger.info(args.note)
    torch.manual_seed(args.seed)

    # mean and std
    humanml_mean = torch.from_numpy(np.load('dataset/HumanML3D/Mean.npy')).cuda()[None, None, ...] # dataset/HumanML3D/Mean.npy
    humanml_std = torch.from_numpy(np.load('dataset/HumanML3D/Std.npy')).cuda()[None, None, ...]
    
    # CLIP
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
    # clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
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
    clip_model = TextCLIP(clip_model)

    # VA-VAE
    if args.modeltype == 'omni67':
        from models.omni67 import CMDM
        net = CMDM(args, args.modeltype, njoints=67 if args.dataset_name == 't2m' else 63)
    elif args.modeltype == 'semboost':
        from models.semanticboost import SemanticBoost
        from utils.model_util import get_semanticboost_args
        net = SemanticBoost(**get_semanticboost_args(args))
    else:   
        raise ValueError("modeltype not found")

    from utils.model_util import create_gaussian_diffusion_simple
    diffusion = create_gaussian_diffusion_simple(args, net, args.modeltype, clip_model)

    load_ckpt(net, args.resume_trans, key='trans')
            
    if sys.gettrace():
        net.eval(); logger.info(' net is eval !!!!!!!')
    else:
        net.train(); logger.info(' net is train ~~~~~')

    net = nn.DataParallel(net, device_ids=list(range(0,len(args.gpu))))
    net.cuda()


    # dataloader
    train_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode=args.mode)
    train_loader_iter = dataset_control.cycle(train_loader)
    # val_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode='train', split='val')
    # val_loader_iter = dataset_control.cycle(val_loader)
    test_loader = dataset_control.DataLoader(batch_size=128, args=args, mode='eval', split='test', shuffle=False, num_workers=4, drop_last=True)

    # 训练配置
    optimizer = initial_optim(args.lr, args.weight_decay, net, args.optimizer)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

    # 2个基于VQVAE的根网络
    if args.modeltype == 'omni67':
        diffusion.trainer_func_omni67(train_loader_iter, logger, optimizer, scheduler, test_loader=test_loader)
    elif args.modeltype == 'semboost':
        diffusion.trainer_func_semboost(train_loader_iter, logger, optimizer, scheduler)
    


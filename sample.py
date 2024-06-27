import options.option_transformer as option_trans
import os 
args = option_trans.get_args_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
# os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)
from utils.fixseed import fixseed
from dataset import dataset_control
from utils.mask_utils import generate_src_mask, load_ckpt
from utils.model_util import create_gaussian_diffusion_simple, get_clip_model, sample_ADControl
from utils.text_control_example import collate_all
import clip

if __name__ == '__main__':

    args.stage2_repeat_times = 1
    args.control_joint = 21
    args.density = 100
    ### stage 1
    args.resume_root = 'output/0518_omni67_multi_partxyz/net_last.pth'; args.roottype = 'omni67'; outname = 'omni67' ; args.normalize_traj=True 
    ### stage 2
    args.resume_trans = 'output/0520_semboost/net_last.pth'; args.modeltype = 'semboost'

    clip_model = get_clip_model()

    # 根节点网络
    if args.roottype == 'omni67':
        from models.omni67 import CMDM
        net_root = CMDM(args, args.roottype)
    load_ckpt(net_root, args.resume_root, key='trans')
    net_root.eval()
    net_root.cuda()
    diffusion_root = create_gaussian_diffusion_simple(args, net_root, args.roottype, clip_model)

    # 2阶段网络
    if args.modeltype == 'diffmae_stage2_2':
        from models.diffmae_2 import DiffMAE2
        net = DiffMAE2(dataset=args.dataset_name, args=args, num_layers_E=8, num_layers_D=0)
    elif args.modeltype == 'semboost':
        from models.semanticboost import SemanticBoost
        from utils.model_util import get_semanticboost_args
        net = SemanticBoost(**get_semanticboost_args(args))
    load_ckpt(net, args.resume_trans, key='trans')
    net.eval()
    net.cuda()
    diffusion = create_gaussian_diffusion_simple(args, net, args.modeltype, clip_model)

    # create dataloader
    args.batch_size = 1
    # train_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode='eval', shuffle=False,)
    # train_loader_iter = dataset_control.cycle(train_loader)
    # val_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode='eval', split='val', shuffle=True, num_workers=0)
    # val_loader_iter = dataset_control.cycle(val_loader)
    test_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode='eval', split='test', shuffle=True, num_workers=0, drop_last=True)


    
    for i, batch in enumerate(test_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, gt_motion, real_length, txt_tokens, traj, traj_mask_263, traj_mask = batch
        b, max_length, num_features = gt_motion.shape
        gt_motion = gt_motion.cuda()
        real_length = real_length.cuda()
        traj = traj.cuda()
        traj_mask = traj_mask.cuda()
        traj_mask_263 = traj_mask_263.cuda()
        real_mask = generate_src_mask(max_length, real_length) # (b,196)
        gt_ric = gt_motion[..., :67]

        text = clip.tokenize(clip_text, truncate=True).cuda() 
        text_emb, word_emb = clip_model(text)

        condition = {}
        condition['traj'] = traj.clone()
        condition['traj_mask'] = traj_mask
        condition['gt_ric'] = gt_ric
        condition['traj_mask_263'] = traj_mask_263
        condition['gt_motion'] = gt_motion
        condition['real_mask'] = real_mask
        condition['clip_text'] = clip_text
        condition['text_emb'] = text_emb
        condition['word_emb'] = word_emb

        sample, loss_xyz = sample_ADControl(diffusion_root, diffusion,  args, condition, vis=True)
        print(f'loss_xyz = {loss_xyz.item():.4f}')
        break


        

    

    
    
    
    
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for Amass',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--loss_type', type=str, choices=['l1', 'l2'], default='l1')
    # parser.add_argument("--control_joint", default=-1, type=int, help='-1 means randomly choose a joint')
    parser.add_argument('--control_joint', default=[-1], nargs="+", type=int)
    parser.add_argument('--density',  type=int, default=0)
    parser.add_argument('--dense_control', action='store_true', default=False)
    parser.add_argument('--cross_traj_feat', action='store_true', default=False)
    parser.add_argument('--add_traj_feat', action='store_true', default=False)
    parser.add_argument('--root_mask_noise_all', action='store_true', default=False)
    parser.add_argument('--mask_noise', action='store_true', default=False)
    parser.add_argument('--attn_mask', action='store_true', default=False)
    parser.add_argument('--no_root_control', action='store_true', default=False)
    parser.add_argument('--root_loss_no_vel_rot', action='store_true', default=False, help='no calc 4 value of root in 263')
    parser.add_argument('--root_dist_loss', action='store_true', default=False, help='instead of element-wise loss, use distance loss for root')
    parser.add_argument('--full_traj_control', action='store_true', default=False, help='use full trajectory control when training')
    parser.add_argument('--mode', type=str, choices=['train', 'val', 'debug'], default='train')
    parser.add_argument('--mask_root',  type=float, default=0.0, help='mask root in 263 when training')
    parser.add_argument('--mask_vel',  type=float, default=0.0, help='mask vel in 263 when training')
    parser.add_argument('--random_t2m', type=float, default=0.0, help='add t2m training')
    parser.add_argument('--multi_joint_control', action='store_true', default=False)
    parser.add_argument('--temporal_complete', type=float, default=0.0, help='whether add temporal completion')
    parser.add_argument('--mask_range', default=[0.5, 1.0], nargs="+", type=float, help='the mask range in dataloader')
    parser.add_argument('--save_iter', default=2000, type=int, help='save frequency')
    parser.add_argument('--normalize_traj', action='store_true', default=False)
    parser.add_argument('--commit_weight', type=float, default=0.02)
    parser.add_argument('--loss_xyz', type=float, default=0.0)
    parser.add_argument('--xyz_type', type=str, choices=['all', 'part'], default='all')
    parser.add_argument('--note', type=str, default='this is note')
    parser.add_argument('--num_layers_E', type=int, default=3)
    parser.add_argument('--num_layers_D', type=int, default=1)
    parser.add_argument('--modeltype', type=str, default='ED') 
    # parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--gpu', nargs='+', default=['0'])
    parser.add_argument('--vq_dir', type=str, default='None')
    parser.add_argument('--pearson', action='store_true', default=False)
    parser.add_argument('--traj-supervise', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--no_random_mask', action='store_true', default=False, help='to check if basic code except mask modeling part is correct')
    ## dataloader
    parser.add_argument('--dataname', type=str, default='t2m', help='dataset directory')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--fps', default=[20], nargs="+", type=int, help='frames per second')
    parser.add_argument('--seq-len', type=int, default=64, help='training motion length')
    
    ## optimization
    parser.add_argument('--total-iter', default=300000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[150000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")
    
    parser.add_argument('--weight-decay', default=1e-6, type=float, help='weight decay') 
    parser.add_argument('--decay-option',default='all', type=str, choices=['all', 'noVQ'], help='disable weight decay on codebook')
    parser.add_argument('--optimizer',default='adamw', type=str, choices=['adam', 'adamw'], help='disable weight decay on codebook')
    
    ## vqvae arch
    parser.add_argument("--code-dim", type=int, default=32, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=8192, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')

    ## gpt arch
    parser.add_argument("--block-size", type=int, default=51, help="seq len")
    parser.add_argument("--embed-dim-gpt", type=int, default=1024, help="embedding dimension")
    parser.add_argument("--clip-dim", type=int, default=512, help="latent dimension in the clip feature")
    parser.add_argument("--num-layers", type=int, default=9, help="nb of transformer layers")
    parser.add_argument("--num-local-layer", type=int, default=1, help="nb of transformer local layers")
    parser.add_argument("--n-head-gpt", type=int, default=16, help="nb of heads")
    parser.add_argument("--ff-rate", type=int, default=4, help="feedforward size")
    parser.add_argument("--drop-out-rate", type=float, default=0.1, help="dropout ratio in the pos encoding")
    
    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add_argument('--quantbeta', type=float, default=1.0, help='dataset directory')

    ## resume
    parser.add_argument("--resume-pth", type=str, default=None, help='resume vq pth')
    parser.add_argument("--resume-trans", type=str, default=None, help='resume gpt pth')
    
    
    ## output directory 
    parser.add_argument('--out-dir', type=str, default='output', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out-dir')
    parser.add_argument('--vq-name', type=str, default='exp_debug', help='name of the generated dataset .npy, will create a file inside out-dir')
    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=20000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training. ')
    parser.add_argument("--if-maxtest", action='store_true', help="test in max")
    parser.add_argument('--pkeep', type=float, default=.5, help='keep rate for gpt training')
    
    ## generator
    parser.add_argument('--text', type=str, help='text')
    parser.add_argument('--length', type=int, help='length')

    return parser.parse_args()
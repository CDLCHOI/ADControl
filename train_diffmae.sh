##### 训练 DiffMAE2 基本仿照mdm的   可用 √
# 0519 用semanticboost作为2阶段
    OMP_NUM_THREADS=8 python train_trans_ED.py --exp-name 0519_semboost --batch-size 128 --gpu 3 --overwrite --print-iter 20 --save_iter 10000 --total-iter 300000 --lr 2e-4 --lr-scheduler 100000 --modeltype semboost --loss_xyz 1 --xyz_type part --loss_type l2 --multi_joint_control --root_dist_loss --resume-trans /data/motion/_files/semanticboost/weights/llama_decoder_norm.pth
# 0520 0519的半夜断了
    OMP_NUM_THREADS=8 python train_trans_ED.py --exp-name 0520_semboost --batch-size 128 --gpu 3 --overwrite --print-iter 20 --save_iter 10000 --total-iter 250000 --lr 2e-4 --lr-scheduler 50000 --modeltype semboost --loss_xyz 1 --xyz_type part --loss_type l2 --multi_joint_control --root_dist_loss --resume-trans output/0519_semboost/net_last.pth

##### 用DiffMAE2训练1阶段网络  diffmae_root67
# omni67
OMP_NUM_THREADS=8 python train_trans_ED.py --exp-name 0518_omni67_multi_partxyz --batch-size 128 --gpu 3 --overwrite --print-iter 20 --save_iter 10000 --total-iter 200000 --lr 2e-4 --lr-scheduler 30000 --modeltype omni67 --loss_xyz 1 --xyz_type part --normalize_traj --multi_joint_control --root_dist_loss  --resume-trans output/0517_omni67_xyzpart/net_last.pth
# 0529
OMP_NUM_THREADS=8 python train_trans_ED.py --exp-name 0529_omni67 --batch-size 128 --gpu 3 --overwrite --print-iter 20 --save_iter 10000 --total-iter 200000 --lr 2e-5 --lr-scheduler 30000 --modeltype omni67 --loss_xyz 1 --xyz_type part --normalize_traj --multi_joint_control --root_dist_loss  --resume-trans output/0518_omni67_multi_partxyz/net_last.pth
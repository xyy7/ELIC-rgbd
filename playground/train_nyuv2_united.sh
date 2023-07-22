# python train2united.py -m ELIC_cpf --epochs 400 -lr 1e-4 --save --git --gpu_id 0 -q 2_2 --restore &
# python train2united.py -m ELIC_cpf --epochs 400 -lr 1e-4 --save  --gpu_id 2 -q 3_3 --restore &
# python train2united.py -m ELIC_cpf --epochs 400 -lr 1e-4 --save  --gpu_id 4 -q 4_4 --restore &
# python train2united.py -m ELIC_cpf --epochs 400 -lr 1e-4 --save  --gpu_id 6 -q 5_5 --restore &


python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6666 train2united.py -m ELIC_cpf --epochs 200 -lr 1e-4 --save --gpu_id 7,2 -q 2_2 --restore --dist&
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6669 train2united.py -m ELIC_cpf --epochs 200 -lr 1e-4 --save  --gpu_id 3,4 -q 3_3 --restore --dist&
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6670 train2united.py -m ELIC_cpf --epochs 200 -lr 1e-4 --save  --gpu_id 5,6 -q 4_4 --restore --dist&
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6667 train2united.py -m ELIC_cpf --epochs 200 -lr 1e-4 --save  --gpu_id 0,1 -q 5_5 --restore --dist&
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6669 train2united.py -m ELIC_cpf --epochs 100 -lr 1e-4 --save  --gpu_id 6,7 -q 5_5 --restore --dist&
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6669 train2united.py -m ELIC_cpf1 --epochs 600 -lr 1e-4 --save  --gpu_id 6,7 -q 5_5 --restore --dist&
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6669 train2united.py -m ELIC_cpf --epochs 400 -lr 1e-4 --save  --gpu_id 6,7 -q 2_2 --restore --dist&
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6669 train2united.py -m ELIC_cpf --epochs 400 -lr 1e-4 --save  --gpu_id 6,7 -q 3_2 --restore --dist&
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6669 train2united.py -m ELIC_cpf --epochs 400 -lr 1e-4 --save  --gpu_id 6,7 -q 3_4 --restore --dist&
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6669 train2united.py -m ELIC_cpf --epochs 400 -lr 1e-4 --save  --gpu_id 6,7 -q 4.5_4.5 --restore --dist&



# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6666 train2united.py -m ELIC_cpf --epochs 400 -lr 1e-4 --save --git --gpu_id 0,1 -q 2_4 --restore --dist&
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6667 train2united.py -m ELIC_cpf --epochs 400 -lr 1e-4 --save  --gpu_id 2,3 -q 3_5 --restore --dist&
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6668 train2united.py -m ELIC_cpf --epochs 400 -lr 1e-4 --save  --gpu_id 4,5 -q 4_6 --restore --dist&
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6669 train2united.py -m ELIC_cpf --epochs 400 -lr 1e-4 --save  --gpu_id 6,7 -q 5_7 --restore --dist&


# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6666 train2united.py -m ELIC_EEM --epochs 600 -lr 1e-4 --save --git --gpu_id 0,1 -q 2_2 --restore --dist&
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6667 train2united.py -m ELIC_EEM --epochs 600 -lr 1e-4 --save  --gpu_id 2,3 -q 2_3 --restore --dist &
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6668 train2united.py -m ELIC_EEM --epochs 600 -lr 1e-4 --save  --gpu_id 4,5 -q 3_5 --restore --dist &
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6669 train2united.py -m ELIC_EEM --epochs 400 -lr 1e-4 --save  --gpu_id 6,7 -q 5_7 --restore --dist &
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6669 train2united.py -m ELIC_EEM --epochs 400 -lr 1e-4 --save  --gpu_id 6,7 -q 3_2 --restore --dist &
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6669 train2united.py -m ELIC_EEM --epochs 400 -lr 1e-4 --save  --gpu_id 6,7 -q 3_4 --restore --dist &
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6669 train2united.py -m ELIC_EEM --epochs 400 -lr 1e-4 --save  --gpu_id 6,7 -q 4_3 --restore --dist &
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6669 train2united.py -m ELIC_EEM --epochs 400 -lr 1e-4 --save  --gpu_id 6,7 -q 4_5 --restore --dist &
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6669 train2united.py -m ELIC_EEM --epochs 400 -lr 1e-4 --save  --gpu_id 6,7 -q 5_5 --restore --dist &


# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6666 train2united.py -m ELIC_united --epochs 600 -lr 1e-4 --save --git --gpu_id 0,1 -q 2_3 --restore --dist &
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6669 train2united.py -m ELIC_united --epochs 600 -lr 1e-4 --save  --gpu_id 6,7 -q 5_6 --restore --dist &
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6667 train2united.py -m ELIC_united --epochs 600 -lr 1e-4 --save  --gpu_id 2,3 -q 3_4 --restore --dist &
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6668 train2united.py -m ELIC_united --epochs 600 -lr 1e-4 --save  --gpu_id 4,5 -q 4_5 --restore --dist &
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6666 train2united.py -m ELIC_united --epochs 400 -lr 1e-4 --save  --gpu_id 0,1 -q 2_2 --restore --dist &
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6666 train2united.py -m ELIC_united --epochs 400 -lr 1e-4 --save  --gpu_id 0,1 -q 2_1 --restore --dist &
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6669 train2united.py -m ELIC_united --epochs 400 -lr 1e-4 --save  --gpu_id 6,7 -q 3_2 --restore --dist &
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6669 train2united.py -m ELIC_united --epochs 400 -lr 1e-4 --save  --gpu_id 6,7 -q 5_4 --restore --dist &
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6667 train2united.py -m ELIC_united --epochs 400 -lr 1e-4 --save  --gpu_id 2,3 -q 3_3 --restore --dist &
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6668 train2united.py -m ELIC_united --epochs 400 -lr 1e-4 --save  --gpu_id 4,5 -q 4_3 --restore --dist &
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6668 train2united.py -m ELIC_united --epochs 400 -lr 1e-4 --save  --gpu_id 4,5 -q 4_4 --restore --dist &
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6668 train2united.py -m ELIC_united --epochs 400 -lr 1e-4 --save  --gpu_id 4,5 -q 4.5_4.5 --restore --dist &
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6668 train2united.py -m ELIC_united --epochs 400 -lr 1e-4 --save  --gpu_id 4,5 -q 5_5 --restore --dist &
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6668 train2united.py -m ELIC_united --epochs 400 -lr 1e-4 --save  --gpu_id 4,5 -q 5_4 --restore --dist &


# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6668 train2united.py -m ELIC_united1 --epochs 600 -lr 1e-4 --save  --gpu_id 4,5 -q 3_4 --restore --dist --checkpoint /data/chenminghui/ELIC/experiments/nyuv2_ELIC_united_2_2/checkpoints/checkpoint_best_loss.pth.tar &
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6668 train2united.py -m ELIC_united1 --epochs 700 -lr 1e-4 --save  --gpu_id 4,5 -q 2_2.5 --restore --dist --checkpoint /data/chenminghui/ELIC/experiments/nyuv2_ELIC_united_2_2/checkpoints/checkpoint_best_loss.pth.tar &
# # python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6668 train2united.py -m ELIC_united --epochs 600 -lr 1e-4 --save  --gpu_id 4,5 -q 2_2 --restore --dist &
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6668 train2united.py -m ELIC_united1 --epochs 900 -lr 1e-4 --save  --gpu_id 4,5 -q 5_5 --restore --dist --checkpoint /data/chenminghui/ELIC/experiments/nyuv2_ELIC_united_4_5/checkpoints/checkpoint_best_loss.pth.tar &

# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6668 train2united.py -m ELIC_united2 --epochs 400 -lr 1e-4 --save  --gpu_id 1,2 -q 2_2.5 --restore --dist &
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6669 train2united.py -m ELIC_united --epochs 400 -lr 1e-4 --save  --gpu_id 6,7 -q 4_4.5 --restore --dist &
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6677 train2united.py -m ELIC_united --epochs 400 -lr 1e-4 --save  --gpu_id 4,5 -q 5_5.5 --restore --dist &

# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6670 train2united.py -m ELIC_united --epochs 400 -lr 1e-4 --save  --gpu_id 0,3 -q 1_1.5 --restore --dist &

# # python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6668 train2united.py -m ELIC_united --epochs 800 -lr 1e-4 --save  --gpu_id 4,5 -q 3_4 --restore --dist &

# # python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6668 train2united.py -m ELIC_united --epochs 600 -lr 1e-4 --save  --gpu_id 4,5 -q 5_4 --restore --dist &


# python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_port 6669 train2united4.py -m ELIC_united4 --epochs 400 -lr 1e-4 --save  --gpu_id 4,5,6,7 -q 2_2.5 --restore --dist&

# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6670 train2united.py -m ELIC_cpf2 --epochs 500 -lr 1e-4 --save  --gpu_id 0,3 -q 5_5 --restore --dist  --checkpoint /data/chenminghui/ELIC/experiments/nyuv2_ELIC_cpf2_5_5/checkpoints/checkpoint_epoch399.pth.tar &

# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6670 train2united.py -m ELIC_cpf3 --epochs 400 -lr 1e-4 --save  --gpu_id 0,3 -q 5_5 --restore --dist  &

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 7669 train2united4.py -m ELIC_united4C --epochs 400 -lr 1e-4 --save  --gpu_id 1,2 -q 2_2 --restore --dist --batch_size 5&

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 7667 train2united4.py -m ELIC_united4A --epochs 400 -lr 1e-4 --save  --gpu_id 4,5 -q 4_4 --restore --dist&

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 7769 train2united4.py -m ELIC_united4_SE_res --epochs 400 -lr 1e-4 --save  --gpu_id 0,6 -q 4_4 --restore --dist &

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 7779 train2united.py -m ELIC_cpf --epochs 400 -lr 1e-4 --save  --gpu_id 3,7 -q 5_5 --restore --dist&

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 7669 train2united4.py -m ELIC_EEM1 --epochs 400 -lr 1e-4 --save  --gpu_id 1,2 -q 2_2 --restore --dist &
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6789 train2united4.py -m ELIC_EEM_se_plus --epochs 400 -lr 1e-4 --save  --gpu_id 3,7 -q 2_2 --restore --dist

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 7779 train2united4.py -m ELIC_united4_SE_res --epochs 400 -lr 1e-4 --save  --gpu_id 1,2 -q 3_3 --restore --dist

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 8769 train2united4.py -m ELIC_EEM_SE_res --epochs 400 -lr 1e-4 --save  --gpu_id 3,4 -q 4_4 --restore --dist

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 5869 train2united4.py -m ELIC_EEM_SE_res --epochs 400 -lr 1e-4 --save  --gpu_id 1,2 -q 2_2 --dist

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 5678 train2united4.py -m ELIC_EEM_SE_res --epochs 400 -lr 1e-4 --save  --gpu_id 0,6 -q 5_5 --dist

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6665 train2united4.py -m ELIC_united_high --epochs 400 -lr 1e-4 --save  --gpu_id 0,6 -q 4_4 --dist --restore --checkpoint /data/chenminghui/ELIC/experiments/nyuv2_ELIC_united4_SE_res_3_3/checkpoints/checkpoint_epoch199.pth.tar

# 证明bpp是更难优化的,所以先优化bpp,具有一定的效果
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 7967 train2united4.py -m ELIC_united_lh --epochs 400 -lr 1e-4 --save  --gpu_id 0,7 -q 3_3.25 --dist --restore --checkpoint /data/chenminghui/ELIC/experiments/nyuv2_ELIC_united4_SE_res_3_3/checkpoints/checkpoint_epoch199.pth.tar

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 7968 train2united4.py -m ELIC_cpf_lh --epochs 400 -lr 1e-4 --save  --gpu_id 5,6 -q 3_3.25 --dist --restore --checkpoint /data/chenminghui/ELIC/experiments/nyuv2_ELIC_cpf_3_3/checkpoints/checkpoint_epoch199.pth.tar


python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6965 train2united4.py -m ELIC_united_lh --epochs 400 -lr 1e-4 --save  --gpu_id 0,3 -q 3_3.5 --dist --restore --checkpoint /data/chenminghui/ELIC/experiments/nyuv2_ELIC_united4_SE_res_2_2/checkpoints/checkpoint_epoch199.pth.tar
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6961 train2united4.py -m ELIC_united_lh --epochs 400 -lr 1e-4 --save  --gpu_id 4,5 -q 4_4.5 --dist --restore --checkpoint /data/chenminghui/ELIC/experiments/nyuv2_ELIC_united4_SE_res_3_3/checkpoints/checkpoint_epoch199.pth.tar


python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6963 train2united4.py -m ELIC_united_continue --epochs 500 -lr 1e-4 --save  --gpu_id 3,4 -q 4_4 --dist --restore --checkpoint /data/chenminghui/ELIC/experiments/nyuv2_ELIC_united4_SE_res_4_4/checkpoints/checkpoint_best_loss.pth.tar


python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6963 train2united-sun.py -m ELIC_united_lh --epochs 600 -lr 1e-4 --save  --gpu_id 3,4 -q 4_4 --dist --restore --checkpoint /data/chenminghui/ELIC/experiments/nyuv2_ELIC_united4_SE_res_4_4/checkpoints/checkpoint_epoch399.pth.tar

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6789 train2united-sun.py -m ELIC_united_lh --epochs 600 -lr 1e-4 --save  --gpu_id 5,6 -q 5_5 --dist --restore --checkpoint /data/chenminghui/ELIC/experiments/nyuv2_ELIC_united4_SE_res_4_4/checkpoints/checkpoint_epoch399.pth.tar


python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6963 train2united-sun.py -m ELIC_united_lh --epochs 600 -lr 1e-4 --save  --gpu_id 4,5 -q 2_2 --dist --restore --checkpoint /data/chenminghui/ELIC/experiments/nyuv2_ELIC_united4_SE_res_2_2/checkpoints/checkpoint_epoch399.pth.tar

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6789 train2united-sun.py -m ELIC_united_lh --epochs 600 -lr 1e-4 --save  --gpu_id 6,7 -q 3_3 --dist --restore --checkpoint /data/chenminghui/ELIC/experiments/nyuv2_ELIC_united4_SE_res_3_3/checkpoints/checkpoint_epoch399.pth.tar



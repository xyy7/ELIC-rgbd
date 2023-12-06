# CUDA_VISIBLE_DEVICES=4 python test_master-sun.py --split depth --gpu_id 4 --q 1 -m ELIC_master --checkpoint /home/xyy/ELIC/experiments/sunrgbd_depth_ELIC_master_1/checkpoints/checkpoint_epoch599.pth.tar  --checkpoint1 /home/xyy/ELIC/experiments/sunrgbd_rgb_ELIC_2/checkpoints/checkpoint_epoch599.pth.tar & 

# CUDA_VISIBLE_DEVICES=5 python test_master-sun.py --split depth  --gpu_id 5 --q 2 -m ELIC_master --checkpoint /home/xyy/ELIC/experiments/sunrgbd_depth_ELIC_master_2/checkpoints/checkpoint_epoch599.pth.tar  --checkpoint1 /home/xyy/ELIC/experiments/sunrgbd_rgb_ELIC_3/checkpoints/checkpoint_epoch599.pth.tar &

# CUDA_VISIBLE_DEVICES=6 python test_master-sun.py --split depth  --gpu_id 6 --q 3 -m ELIC_master --checkpoint /home/xyy/ELIC/experiments/sunrgbd_depth_ELIC_master_3/checkpoints/checkpoint_epoch599.pth.tar  --checkpoint1 /home/xyy/ELIC/experiments/sunrgbd_rgb_ELIC_4/checkpoints/checkpoint_epoch599.pth.tar &

# CUDA_VISIBLE_DEVICES=7 python test_master-sun.py --split depth  --gpu_id 7 --q 4 -m ELIC_master --checkpoint /home/xyy/ELIC/experiments/sunrgbd_depth_ELIC_master_4/checkpoints/checkpoint_epoch599.pth.tar  --checkpoint1 /home/xyy/ELIC/experiments/sunrgbd_rgb_ELIC_5/checkpoints/checkpoint_epoch599.pth.tar &


CUDA_VISIBLE_DEVICES=2 python test_united4-sun.py  --gpu_id 2 --q 2_2 -m ELIC_united_mse &
CUDA_VISIBLE_DEVICES=6 python test_united4-sun.py   --gpu_id 6 --q 3_3 -m ELIC_united_mse &
CUDA_VISIBLE_DEVICES=2 python test_united4-sun.py  --gpu_id 6 --q 4_4 -m ELIC_united_mse &
CUDA_VISIBLE_DEVICES=6 python test_united4-sun.py   --gpu_id 7 --q 5_5 -m ELIC_united_mse &
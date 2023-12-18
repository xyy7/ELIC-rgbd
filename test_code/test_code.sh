cd ../playground
# python train.py --save --gpu_id 6 -q 2_2 -e 2 -m ELIC_united --channel 4 --dataset /data/xyy/sunrgbd/train/train --val_dataset /data/xyy/sunrgbd/train/val --debug
# python train.py --save --gpu_id 6 -q 2_2 -e 2 -m ELIC_united --channel 4 --dataset "/data/xyy/nyu5k" --val_dataset "/data/xyy/nyu5k/val" --debug

# python train.py --save --gpu_id 6 -q 2_2 -e 2 -m ELIC_united_CPT --channel 4 --dataset /data/xyy/sunrgbd/train/train --val_dataset /data/xyy/sunrgbd/train/val --debug
# python train.py --save --gpu_id 6 -q 2_2 -e 2 -m ELIC_united_CPT --channel 4 --dataset "/data/xyy/nyu5k" --val_dataset "/data/xyy/nyu5k/val" --debug

# python train.py --save --gpu_id 6 -q 2_2 -e 2 -m ELIC_united_CCE --channel 4 --dataset /data/xyy/sunrgbd/train/train --val_dataset /data/xyy/sunrgbd/train/val --debug
# python train.py --save --gpu_id 6 -q 2_2 -e 2 -m ELIC_united_CCE --channel 4 --dataset "/data/xyy/nyu5k" --val_dataset "/data/xyy/nyu5k/val" --debug

# python train.py --save --gpu_id 6 -q 2_2 -e 2 -m ELIC_united_R2D --channel 4 --dataset /data/xyy/sunrgbd/train/train --val_dataset /data/xyy/sunrgbd/train/val --debug
# python train.py --save --gpu_id 6 -q 2_2 -e 2 -m ELIC_united_R2D --channel 4 --dataset "/data/xyy/nyu5k" --val_dataset "/data/xyy/nyu5k/val" --debug

# python train.py --save --gpu_id 6 -q 2 -e 2 -m ELIC --channel 3 --dataset /data/xyy/sunrgbd/train/train --val_dataset /data/xyy/sunrgbd/train/val --debug
# python train.py --save --gpu_id 6 -q 2 -e 2 -m ELIC --channel 3 --dataset "/data/xyy/nyu5k" --val_dataset "/data/xyy/nyu5k/val" --debug

# python train.py --save --gpu_id 6 -q 2 -e 2 -m ELIC --channel 1 --dataset /data/xyy/sunrgbd/train/train --val_dataset /data/xyy/sunrgbd/train/val --debug
# python train.py --save --gpu_id 6 -q 2 -e 2 -m ELIC --channel 1 --dataset "/data/xyy/nyu5k" --val_dataset "/data/xyy/nyu5k/val" --debug

# python train.py --save --gpu_id 6 -q 2 -e 2 -m ELIC_master --channel 3 --dataset /data/xyy/sunrgbd/train/train --val_dataset /data/xyy/sunrgbd/train/val --debug -c1 /home/xyy/ELIC/experiments_test/sunrgbd_depth_ELIC_2/checkpoints/checkpoint_best_loss.pth.tar
# python train.py --save --gpu_id 6 -q 2 -e 2 -m ELIC_master --channel 3 --dataset "/data/xyy/nyu5k" --val_dataset "/data/xyy/nyu5k/val" --debug -c1 /home/xyy/ELIC/experiments_test/nyuv2_depth_ELIC_2/checkpoints/checkpoint_best_loss.pth.tar 

python train.py --save --gpu_id 6 -q 2 -e 2 -m ELIC_master --channel 1 --dataset /data/xyy/sunrgbd/train/train --val_dataset /data/xyy/sunrgbd/train/val --debug -c1 /home/xyy/ELIC/experiments_test/sunrgbd_rgb_ELIC_2/checkpoints/checkpoint_best_loss.pth.tar
python train.py --save --gpu_id 6 -q 2 -e 2 -m ELIC_master --channel 1 --dataset "/data/xyy/nyu5k" --val_dataset "/data/xyy/nyu5k/val" --debug -c1 /home/xyy/ELIC/experiments_test/nyuv2_rgb_ELIC_2/checkpoints/checkpoint_best_loss.pth.tar
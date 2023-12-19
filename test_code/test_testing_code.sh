cd ../playground
## test united
# python test.py --channel 4 -m ELIC_united --gpu_id 6 -q 2_2 --debug --dataset /data/xyy/nyu5k/nyuv2/test
# python test.py --channel 4 -m ELIC_united --gpu_id 6 -q 2_2 --debug --dataset /data/xyy/sunrgbd/test

# python test.py --channel 4 -m ELIC_united_CPT --gpu_id 6 -q 2_2 --debug --dataset /data/xyy/nyu5k/nyuv2/test
# python test.py --channel 4 -m ELIC_united_CPT --gpu_id 6 -q 2_2 --debug --dataset /data/xyy/sunrgbd/test

# python test.py --channel 4 -m ELIC_united_CCE --gpu_id 6 -q 2_2 --debug --dataset /data/xyy/nyu5k/nyuv2/test
# python test.py --channel 4 -m ELIC_united_CCE --gpu_id 6 -q 2_2 --debug --dataset /data/xyy/sunrgbd/test

# python test.py --channel 4 -m ELIC_united_R2D --gpu_id 6 -q 2_2 --debug --dataset /data/xyy/nyu5k/nyuv2/test
# python test.py --channel 4 -m ELIC_united_R2D --gpu_id 6 -q 2_2 --debug --dataset /data/xyy/sunrgbd/test

## test single
# python test.py --channel 3 -m ELIC --gpu_id 6 -q 2 --debug --dataset /data/xyy/nyu5k/nyuv2/test
# python test.py --channel 3 -m ELIC --gpu_id 6 -q 2 --debug --dataset /data/xyy/sunrgbd/test

# python test.py --channel 1 -m ELIC --gpu_id 6 -q 2 --debug --dataset /data/xyy/nyu5k/nyuv2/test
# python test.py --channel 1 -m ELIC --gpu_id 6 -q 2 --debug --dataset /data/xyy/sunrgbd/test


## test master
# python test.py --channel 3 -m ELIC_master --gpu_id 6 -q 2 --debug --dataset /data/xyy/nyu5k/nyuv2/test -c1 /home/xyy/ELIC/experiments_test/nyuv2_depth_ELIC_2/checkpoints/checkpoint_best_loss.pth.tar  
# python test.py --channel 3 -m ELIC_master --gpu_id 6 -q 2 --debug --dataset /data/xyy/sunrgbd/test -c1 /home/xyy/ELIC/experiments_test/sunrgbd_depth_ELIC_2/checkpoints/checkpoint_best_loss.pth.tar

python test.py --channel 1 -m ELIC_master --gpu_id 6 -q 2 --debug --dataset /data/xyy/nyu5k/nyuv2/test -c1 /home/xyy/ELIC/experiments_test/nyuv2_rgb_ELIC_2/checkpoints/checkpoint_best_loss.pth.tar 
python test.py --channel 1 -m ELIC_master --gpu_id 6 -q 2 --debug --dataset /data/xyy/sunrgbd/test -c1 /home/xyy/ELIC/experiments_test/sunrgbd_rgb_ELIC_2/checkpoints/checkpoint_best_loss.pth.tar



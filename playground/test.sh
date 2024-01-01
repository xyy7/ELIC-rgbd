## test master
# python test.py --chanel 1  --gpu_id 1 --q 3 -m ELIC_master --checkpoint /home/xyy/ELIC/experiments/nyuv2_depth_ELIC_master-60_3/checkpoints/checkpoint_latest.pth.tar  --checkpoint1 /home/xyy/ELIC/experiments/nyuv2_rgb_ELIC_4/checkpoints/checkpoint_epoch399.pth.tar &

## test united
# python test.py --channel 4 -m ELIC_united_wo_ssim --gpu_id 7 -q 2_2 --dataset /data/xyy/nyu5k/nyuv2/test &
# python test.py --channel 4 -m ELIC_united_wo_ssim --gpu_id 6 -q 3_3 --dataset /data/xyy/nyu5k/nyuv2/test 
# python test.py --channel 4 -m ELIC_united_wo_ssim --gpu_id 7 -q 4_4 --dataset /data/xyy/nyu5k/nyuv2/test &
# python test.py --channel 4 -m ELIC_united_wo_ssim --gpu_id 6 -q 5_5 --dataset /data/xyy/nyu5k/nyuv2/test


# python test.py --channel 4 -m ELIC_united_wo_edge --gpu_id 3 -q 2_2 --dataset /data/xyy/nyu5k/nyuv2/test &
# python test.py --channel 4 -m ELIC_united_wo_edge --gpu_id 4 -q 3_3 --dataset /data/xyy/nyu5k/nyuv2/test &
# python test.py --channel 4 -m ELIC_united_wo_edge --gpu_id 5 -q 4_4 --dataset /data/xyy/nyu5k/nyuv2/test &
# python test.py --channel 4 -m ELIC_united_wo_edge --gpu_id 6 -q 5_5 --dataset /data/xyy/nyu5k/nyuv2/test &

# python test.py --channel 4 -m ELIC_united_wo_edge --gpu_id 3 -q 2_1.5 --dataset /data/xyy/nyu5k/nyuv2/test &
# python test.py --channel 4 -m ELIC_united_wo_edge --gpu_id 4 -q 3_2.5 --dataset /data/xyy/nyu5k/nyuv2/test &
# python test.py --channel 4 -m ELIC_united_wo_edge --gpu_id 5 -q 4_3.5 --dataset /data/xyy/nyu5k/nyuv2/test &
# python test.py --channel 4 -m ELIC_united_wo_edge --gpu_id 6 -q 5_4.5 --dataset /data/xyy/nyu5k/nyuv2/test 

# python test.py --channel 4 -m ELIC_united_wo_edge --gpu_id 3 -q 2_2.5 --dataset /data/xyy/nyu5k/nyuv2/test &
# python test.py --channel 4 -m ELIC_united_wo_edge --gpu_id 4 -q 3_3.5 --dataset /data/xyy/nyu5k/nyuv2/test &
# python test.py --channel 4 -m ELIC_united_wo_edge --gpu_id 5 -q 4_4.5 --dataset /data/xyy/nyu5k/nyuv2/test &
# python test.py --channel 4 -m ELIC_united_wo_edge --gpu_id 6 -q 5_5.5 --dataset /data/xyy/nyu5k/nyuv2/test 


# python test.py --channel 4 -m ELIC_united_wo_l1 --gpu_id 5 -q 2_1 --dataset /data/xyy/nyu5k/nyuv2/test &
# python test.py --channel 4 -m ELIC_united_wo_l1 --gpu_id 4 -q 3_2 --dataset /data/xyy/nyu5k/nyuv2/test &
# python test.py --channel 4 -m ELIC_united_wo_l1 --gpu_id 5 -q 4_3 --dataset /data/xyy/nyu5k/nyuv2/test &
# python test.py --channel 4 -m ELIC_united_wo_l1 --gpu_id 4 -q 5_4 --dataset /data/xyy/nyu5k/nyuv2/test &

## test_concat 
python test.py --channel 4 -m ELIC_united_wo_edge --gpu_id 6 -q 5_5.5 --dataset /data/xyy/nyu5k/nyuv2/test 
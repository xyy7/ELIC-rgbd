# python train2color.py -m ELIC --epochs 600 -lr 1e-4 --save --git --gpu_id 0 -q 1 --restore &
# python train2color.py -m ELIC --epochs 600 -lr 1e-4 --save --gpu_id 1 -q 2 --restore &
# python train2color.py -m ELIC --epochs 600 -lr 1e-4 --save --gpu_id 2 -q 3 --restore &
# python train2color.py -m ELIC --epochs 600 -lr 1e-4 --save --gpu_id 3 -q 4 --restore &

# python train2depth.py -m ELIC --epochs 600 -lr 1e-4 --save --gpu_id 4 -q 1 --restore &
# python train2depth.py -m ELIC --epochs 600 -lr 1e-4 --save --gpu_id 5 -q 2 --restore &
# python train2depth.py -m ELIC --epochs 600 -lr 1e-4 --save --gpu_id 6 -q 3 --restore &
# python train2depth.py -m ELIC --epochs 600 -lr 1e-4 --save --gpu_id 7 -q 4 --restore &

# python train2depth-sun.py -m ELIC --epochs 600 -lr 1e-4 --save --gpu_id 0 -q 1 --restore --checkpoint  /data/chenminghui/ELIC/experiments/nyuv2_depth_ELIC_1/checkpoints/checkpoint_epoch399.pth.tar &
# python train2depth-sun.py -m ELIC --epochs 600 -lr 1e-4 --save --gpu_id 1 -q 2 --restore  --checkpoint  /data/chenminghui/ELIC/experiments/nyuv2_depth_ELIC_2/checkpoints/checkpoint_epoch399.pth.tar&
# python train2depth-sun.py -m ELIC --epochs 600 -lr 1e-4 --save --gpu_id 2 -q 3 --restore  --checkpoint  /data/chenminghui/ELIC/experiments/nyuv2_depth_ELIC_3/checkpoints/checkpoint_epoch399.pth.tar&
# python train2color-sun.py -m ELIC --epochs 600 -lr 1e-4 --save --gpu_id 3 -q 5 --restore  --checkpoint  /data/chenminghui/ELIC/experiments/nyuv2_rgb_ELIC_5/checkpoints/checkpoint_epoch399.pth.tar

# python train2depth.py -m ELIC --epochs 400 -lr 1e-4 --save --gpu_id 4 -q 1 --restore &
# python train2depth.py -m ELIC --epochs 400 -lr 1e-4 --save --gpu_id 5 -q 2 --restore &
# python train2depth.py -m ELIC --epochs 400 -lr 1e-4 --save --gpu_id 6 -q 3 --restore &
# python train2depth.py -m ELIC --epochs 400 -lr 1e-4 --save --gpu_id 7 -q 4 --restore &

#  python train2color.py -m ELIC_200 --epochs 200 -lr 1e-4 --save --gpu_id 5 -q 5 &

# python train2color-ckbd.py -m ckbd --epochs 400 -lr 1e-4 --save --gpu_id 2 -q 2 &
# python train2color-ckbd.py -m ckbd --epochs 400 -lr 1e-4 --save --gpu_id 3 -q 3 &
# python train2color-ckbd.py -m ckbd --epochs 400 -lr 1e-4 --save --gpu_id 4 -q 4 &
# python train2color-ckbd.py -m ckbd --epochs 400 -lr 1e-4 --save --gpu_id 5 -q 5 &


# python train2depth-ckbd.py -m ckbd --epochs 400 -lr 1e-4 --save --gpu_id 0 -q 2 &
# python train2depth-ckbd.py -m ckbd --epochs 400 -lr 1e-4 --save --gpu_id 6 -q 3 &
# python train2depth-ckbd.py -m ckbd --epochs 400 -lr 1e-4 --save --gpu_id 7 -q 4 &
# python train2depth-ckbd.py -m ckbd --epochs 400 -lr 1e-4 --save --gpu_id 1 -q 1 &

python train2depth_master.py -m ELIC_master --epochs 400 -lr 1e-4 --save  --gpu_id 0,1 -q 1  --restore --checkpoint1  /data/chenminghui/ELIC/experiments/nyuv2_rgb_ELIC_2/checkpoints/checkpoint_epoch399.pth.tar &
python train2depth_master.py -m ELIC_master --epochs 400 -lr 1e-4 --save  --gpu_id 2,3 -q 2  --restore --checkpoint1  /data/chenminghui/ELIC/experiments/nyuv2_rgb_ELIC_3/checkpoints/checkpoint_epoch399.pth.tar &
python train2depth_master.py -m ELIC_master --epochs 400 -lr 1e-4 --save  --gpu_id 4,5 -q 3  --restore --checkpoint1  /data/chenminghui/ELIC/experiments/nyuv2_rgb_ELIC_4/checkpoints/checkpoint_epoch399.pth.tar &
python train2depth_master.py -m ELIC_master --epochs 400 -lr 1e-4 --save  --gpu_id 6,7 -q 4  --restore --checkpoint1  /data/chenminghui/ELIC/experiments/nyuv2_rgb_ELIC_5/checkpoints/checkpoint_epoch399.pth.tar &
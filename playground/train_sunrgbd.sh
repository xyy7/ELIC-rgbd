# python train2depth-sun.py -m ELIC --epochs 600 -lr 1e-4 --save --gpu_id 0 -q 1 --restore --checkpoint  /data/chenminghui/ELIC/experiments/nyuv2_depth_ELIC_1/checkpoints/checkpoint_epoch399.pth.tar &
# python train2depth-sun.py -m ELIC --epochs 600 -lr 1e-4 --save --gpu_id 1 -q 2 --restore  --checkpoint  /data/chenminghui/ELIC/experiments/nyuv2_depth_ELIC_2/checkpoints/checkpoint_epoch399.pth.tar&
# python train2depth-sun.py -m ELIC --epochs 600 -lr 1e-4 --save --gpu_id 2 -q 3 --restore  --checkpoint  /data/chenminghui/ELIC/experiments/nyuv2_depth_ELIC_3/checkpoints/checkpoint_epoch399.pth.tar&
# python train2color-sun.py -m ELIC --epochs 600 -lr 1e-4 --save --gpu_id 3 -q 5 --restore  --checkpoint  /data/chenminghui/ELIC/experiments/nyuv2_rgb_ELIC_5/checkpoints/checkpoint_epoch399.pth.tar


python train2depth-ckbd-sun.py -m ckbd --epochs 600 -lr 1e-4 --save --gpu_id 0 -q 1 --restore --checkpoint  /data/chenminghui/ELIC/experiments/nyuv2_depth_ckbd_1/checkpoints/checkpoint_epoch399.pth.tar &
python train2depth-ckbd-sun.py -m ckbd --epochs 600 -lr 1e-4 --save --gpu_id 1 -q 2 --restore  --checkpoint  /data/chenminghui/ELIC/experiments/nyuv2_depth_ckbd_2/checkpoints/checkpoint_epoch399.pth.tar&
python train2depth-ckbd-sun.py -m ckbd --epochs 600 -lr 1e-4 --save --gpu_id 2 -q 3 --restore  --checkpoint  /data/chenminghui/ELIC/experiments/nyuv2_depth_ckbd_3/checkpoints/checkpoint_epoch399.pth.tar&
python train2depth-ckbd-sun.py -m ckbd --epochs 600 -lr 1e-4 --save --gpu_id 3 -q 4 --restore  --checkpoint  /data/chenminghui/ELIC/experiments/nyuv2_depth_ckbd_4/checkpoints/checkpoint_epoch399.pth.tar

python train2color-ckbd-sun.py -m ckbd --epochs 600 -lr 1e-4 --save --gpu_id 0 -q 2 --restore --checkpoint  /data/chenminghui/ELIC/experiments/nyuv2_rgb_ckbd_2/checkpoints/checkpoint_epoch399.pth.tar &
python train2color-ckbd-sun.py -m ckbd --epochs 600 -lr 1e-4 --save --gpu_id 1 -q 3 --restore  --checkpoint  /data/chenminghui/ELIC/experiments/nyuv2_rgb_ckbd_3/checkpoints/checkpoint_epoch399.pth.tar&
python train2color-ckbd-sun.py -m ckbd --epochs 600 -lr 1e-4 --save --gpu_id 2 -q 4 --restore  --checkpoint  /data/chenminghui/ELIC/experiments/nyuv2_rgb_ckbd_4/checkpoints/checkpoint_epoch399.pth.tar&
python train2color-ckbd-sun.py -m ckbd --epochs 600 -lr 1e-4 --save --gpu_id 3 -q 5 --restore  --checkpoint  /data/chenminghui/ELIC/experiments/nyuv2_rgb_ckbd_5/checkpoints/checkpoint_epoch399.pth.tar


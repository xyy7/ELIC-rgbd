## multi GPU
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6666 train.py -m ELIC_united_single --epochs 400 -lr 1e-4 --save --gpu_id 0,1 -q 2_2 --auto_restore --dist --channel 4 &

## ELIC_united_wo_l1
# python train.py -m ELIC_united_wo_l1 --epochs 400 -lr 1e-4 --save  --gpu_id 4 -q 4_4 --channel 4  --auto_restore&
# python train.py -m ELIC_united_wo_l1 --epochs 400 -lr 1e-4 --save  --gpu_id 5 -q 2_2 --channel 4  --auto_restore&
# python train.py -m ELIC_united_wo_l1 --epochs 400 -lr 1e-4 --save  --gpu_id 3 -q 3_3 --channel 4 &
# python train.py -m ELIC_united_wo_l1 --epochs 400 -lr 1e-4 --save  --gpu_id 6 -q 5_5 --channel 4 &

## ELIC_united_wo_ssim
# python train.py -m ELIC_united_wo_ssim --epochs 400 -lr 1e-4 --save  --gpu_id 4 -q 4_4 --channel 4 &
# python train.py -m ELIC_united_wo_ssim --epochs 400 -lr 1e-4 --save  --gpu_id 5 -q 2_2 --channel 4 &
# python train.py -m ELIC_united_wo_ssim --epochs 400 -lr 1e-4 --save  --gpu_id 6 -q 3_3 --channel 4 &
# python train.py -m ELIC_united_wo_ssim --epochs 400 -lr 1e-4 --save  --gpu_id 7 -q 5_5 --channel 4 &

## ELIC_united_wo_edge_loss
# python train.py -m ELIC_united_wo_edge --epochs 400 -lr 1e-4 --save  --gpu_id 4 -q 4_4.5 --channel 4 --start_epoch 340 -c /home/xyy/ELIC/experiments/nyuv2_ELIC_united_wo_edge_4_5/checkpoints/checkpoint_epoch399.pth.tar &
# python train.py -m ELIC_united_wo_edge --epochs 400 -lr 1e-4 --save  --gpu_id 5 -q 2_2.5 --channel 4 --start_epoch 340 -c /home/xyy/ELIC/experiments/nyuv2_ELIC_united_wo_edge_2_3/checkpoints/checkpoint_epoch399.pth.tar&
# python train.py -m ELIC_united_wo_edge --epochs 400 -lr 1e-4 --save  --gpu_id 6 -q 3_3.5 --channel 4 --start_epoch 340 -c /home/xyy/ELIC/experiments/nyuv2_ELIC_united_wo_edge_3_4/checkpoints/checkpoint_epoch399.pth.tar&
# python train.py -m ELIC_united_wo_edge --epochs 400 -lr 1e-4 --save  --gpu_id 7 -q 5_5.5 --channel 4 --start_epoch 340 -c /home/xyy/ELIC/experiments/nyuv2_ELIC_united_wo_edge_5_6/checkpoints/checkpoint_epoch399.pth.tar 

# python train.py -m ELIC_united_wo_edge --epochs 400 -lr 1e-4 --save  --gpu_id 4 -q 4_3.5 --channel 4 --start_epoch 340 -c /home/xyy/ELIC/experiments/nyuv2_ELIC_united_wo_edge_4_5/checkpoints/checkpoint_epoch399.pth.tar &
# python train.py -m ELIC_united_wo_edge --epochs 400 -lr 1e-4 --save  --gpu_id 5 -q 2_1.5 --channel 4 --start_epoch 340 -c /home/xyy/ELIC/experiments/nyuv2_ELIC_united_wo_edge_2_3/checkpoints/checkpoint_epoch399.pth.tar&
# python train.py -m ELIC_united_wo_edge --epochs 400 -lr 1e-4 --save  --gpu_id 6 -q 3_2.5 --channel 4 --start_epoch 340 -c /home/xyy/ELIC/experiments/nyuv2_ELIC_united_wo_edge_3_4/checkpoints/checkpoint_epoch399.pth.tar&
# python train.py -m ELIC_united_wo_edge --epochs 400 -lr 1e-4 --save  --gpu_id 7 -q 5_4.5 --channel 4 --start_epoch 340 -c /home/xyy/ELIC/experiments/nyuv2_ELIC_united_wo_edge_5_6/checkpoints/checkpoint_epoch399.pth.tar 

## ELIC concat
python train.py --save --gpu_id 4 -q 2 -e 400 -lr 1e-4 --save  -m ELIC_cat --channel 4 --batch-size 24&
python train.py --save --gpu_id 5 -q 3 -e 400 -lr 1e-4 --save  -m ELIC_cat --channel 4 --batch-size 24&
python train.py --save --gpu_id 6 -q 4 -e 400 -lr 1e-4 --save  -m ELIC_cat --channel 4 --batch-size 24&
python train.py --save --gpu_id 7 -q 5 -e 400 -lr 1e-4 --save  -m ELIC_cat --channel 4 --batch-size 24&

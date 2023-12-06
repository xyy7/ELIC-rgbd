## multi GPU
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 6666 train.py -m ELIC_united_single --epochs 400 -lr 1e-4 --save --gpu_id 0,1 -q 2_2 --auto_restore --dist --channel 4 &

## ELIC_united_wo_l1
python train.py -m ELIC_united_wo_l1 --epochs 400 -lr 1e-4 --save  --gpu_id 0 -q 4_4 --channel 4 &
python train.py -m ELIC_united_wo_l1 --epochs 400 -lr 1e-4 --save  --gpu_id 0 -q 2_2 --channel 4 &
python train.py -m ELIC_united_wo_l1 --epochs 400 -lr 1e-4 --save  --gpu_id 3 -q 3_3 --channel 4 &
python train.py -m ELIC_united_wo_l1 --epochs 400 -lr 1e-4 --save  --gpu_id 6 -q 5_5 --channel 4 &

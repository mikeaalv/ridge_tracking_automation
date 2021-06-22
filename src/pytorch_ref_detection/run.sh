# recources
srun --pty -p gpu_p --gres=gpu:P100:1 --mem=20gb --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --time=12:00:00 --job-name=qlogin /bin/bash -l
# env
ml load PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4
ml load torchvision/0.7.0-fosscuda-2019b-Python-3.7.4-PyTorch-1.6.0
conda activate pytorch_segmentation
#
python train.py --dataset coco --model maskrcnn_resnet50_fpn --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3 --data-path './data/worm_classify/'

python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py --dataset coco --model maskrcnn_resnet50_fpn --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3 --data-path './data/worm_classify/' --world-size 1

# python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py --dataset coco --model maskrcnn_resnet50_fpn --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

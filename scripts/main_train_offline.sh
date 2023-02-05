export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:`pwd`
python main_train_offline.py \
  --dataset cifar10 \
  --backbone moco_v2 \
  --model lr_bgfs \
  --C 1 \
  --repeat 5 \
  --minmax

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:`pwd`
python main_train_offline.py \
  --dataset cifar10 \
  --backbone moco_v2 \
  --model nb_diag \
  --repeat 5 \
  --minmax
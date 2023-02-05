export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:`pwd`
python main_extract_features.py \
  --dataset cifar10 \
  --backbone moco_v2 \
  --bs 100 \
  --gpu 1 \

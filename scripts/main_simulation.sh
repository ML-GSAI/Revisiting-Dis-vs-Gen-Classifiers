export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:`pwd`
n_list=(100 200 400 1000)
K_list=(2 3 5 7)
time_list=(1 2 3 4 5)
for K in ${K_list[@]};do
for n in ${n_list[@]};do
for t in ${time_list[@]};do
python main_simulation.py \
  --K $K \
  --n $n \
  --t $t \
  --model nb_diag \
  --repeat 1

python main_simulation.py \
  --K $K \
  --n $n \
  --t $t \
  --model lr_bgfs \
  --C 1 \
  --repeat 1
done
done
done
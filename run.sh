dataset=$1
gpu=$2
anomaly_ratio_lst=(0.1 0.05 0.01)


if [[ "$dataset" == "uci" ]]; 
then
for anomaly_ratio in "${anomaly_ratio_lst[@]}"; do
    CUDA_VISIBLE_DEVICES=$gpu python main.py --dataset $dataset --snap_size 1000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio --noise_ratio 0 \
    --ae_epoch 200 --ae_lr 0.0005 --es_epoch 200 --es_lr 0.001 --ad_epoch 800 --ad_lr 0.001 \
    --x_dim 128 --h_dim 128 --z_dim 128 --expand 2
done

elif [[ "$dataset" == "digg" ]]; 
then
for anomaly_ratio in "${anomaly_ratio_lst[@]}"; do
    CUDA_VISIBLE_DEVICES=$gpu python main.py --dataset $dataset --snap_size 6000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio --noise_ratio 0 \
    --ae_epoch 200 --ae_lr 0.0005 --es_epoch 200 --es_lr 0.0005 --ad_epoch 1600 --ad_lr 0.001 \
    --x_dim 128 --h_dim 128 --z_dim 128 --expand 2
done

elif [[ "$dataset" == "btc_otc" ]]; 
then
for anomaly_ratio in "${anomaly_ratio_lst[@]}"; do
    CUDA_VISIBLE_DEVICES=$gpu python main.py --dataset $dataset --snap_size 1000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio --noise_ratio 0 \
    --ae_epoch 100 --ae_lr 0.0001 --es_epoch 150 --es_lr 0.001 --ad_epoch 1600 --ad_lr 0.0005 \
    --x_dim 128 --h_dim 128 --z_dim 128 --expand 2
done

elif [[ "$dataset" == "btc_alpha" ]]; 
then
for anomaly_ratio in "${anomaly_ratio_lst[@]}"; do
    CUDA_VISIBLE_DEVICES=$gpu python main.py --dataset $dataset --snap_size 2000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio --noise_ratio 0 \
    --ae_epoch 100 --ae_lr 0.0005 --es_epoch 200 --es_lr 0.001 --ad_epoch 800 --ad_lr 0.001 \
    --x_dim 128 --h_dim 128 --z_dim 128 --expand 2
done
fi
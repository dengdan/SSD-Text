set -x
set -e
export CUDA_VISIBLE_DEVICES=$1
ACTION=$2
HOME=/home/dengdan
TRAIN_DIR=$HOME/temp/ssd-text-$SIZE/origin-config
SIZE=512
MODEL_NAME=ssd_${SIZE}_vgg

case $ACTION in 
    train)
        DATASET=$HOME/dataset/SSD-tf/ICDAR
        python train_ssd_network.py \
            --dataset_dir=$DATASET \
            --negative_ratio=3 \
            --match_threshold=0.5 \
            --train_dir=$TRAIN_DIR \
            --learning_rate_decay_type=fixed \
            --learning_rate=0.0001 \
            --dataset_name=icdar2013 \
            --dataset_split_name=train \
            --model_name=$MODEL_NAME \
            --batch_size=16 \
            --max_number_of_steps=400000
    ;;
    
    val)
        echo val
    ;;
    
    test)
    
    ;;
    
esac

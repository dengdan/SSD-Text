set -x
set -e
export CUDA_VISIBLE_DEVICES=$1
ACTION=$2
HOME=/home/dengdan
SIZE=512
TRAIN_DIR=$HOME/temp/ssd-text-$SIZE/origin-config
CKPT_PATH=$TRAIN_DIR
MODEL_NAME=ssd_${SIZE}_vgg
DATASET=$HOME/dataset/SSD-tf/ICDAR
EVAL_DIR=${TRAIN_DIR}/eval
case $ACTION in 
    train)
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
            --batch_size=8 \
            --max_number_of_steps=400000
    ;;
    
    eval)
        python eval_ssd_network.py \
            --dataset_dir=$DATASET \
            --checkpoint_path=$CKPT_PATH \
            --eval_dir=$EVAL_DIR\
            --model_name=$MODEL_NAME
    ;;
    
    test)
    
    ;;
    
esac

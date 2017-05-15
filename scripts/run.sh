set -x
set -e
export CUDA_VISIBLE_DEVICES=$1
ACTION=$2
IMG_PER_GPU=$3
SPLIT=$4

# get the number of gpus
OLD_IFS="$IFS" 
IFS="," 
gpus=($CUDA_VISIBLE_DEVICES) 
IFS="$OLD_IFS"
num_gpus=${#gpus[@]}

# batch_size = num_gpus * IMG_PER_GPU
BATCH_SIZE=`expr $num_gpus \* $IMG_PER_GPU`

HOME=/home/dengdan
SIZE=512
TRAIN_DIR=$HOME/temp/ssd-text-$SIZE/SynthText-pretrain/anchor4810
CKPT_PATH=$TRAIN_DIR
MODEL_NAME=ssd_${SIZE}_vgg
EVAL_DIR=${TRAIN_DIR}/eval
case $ACTION in 
    pretrain)
        
        
        DATASET=$HOME/dataset/SSD-tf/SynthText
        python train_ssd_network.py \
            --dataset_dir=$DATASET \
            --negative_ratio=3 \
            --match_threshold=0.5 \
            --train_dir=$TRAIN_DIR \
            --learning_rate_decay_type=fixed \
            --learning_rate=0.0001 \
            --dataset_name=synthtext \ #icdar2013 \
            --dataset_split_name=train \
            --model_name=$MODEL_NAME \
            --batch_size=$BATCH_SIZE \
            --should_trace=0 \
            --gpu_memory_fraction=0.5 \
            --max_number_of_steps=400000 #50000
    ;;
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
            --batch_size=$BATCH_SIZE \
            --should_trace=0 \
            --gpu_memory_fraction=.5 \
            --max_number_of_steps=400000
    ;;
    eval)
        #TRAIN_DIR=$HOME/temp/ssd-text-$SIZE/SynthText-pretrain-cnt/origin-config
        CKPT_PATH=$TRAIN_DIR
        MODEL_NAME=ssd_${SIZE}_vgg
        EVAL_DIR=${TRAIN_DIR}/eval/$SPLIT
        #CUDA_VISIBLE_DEVICES=
        DATASET=$HOME/dataset/SSD-tf/ICDAR
        python eval_ssd_network.py \
            --dataset_dir=$DATASET \
            --checkpoint_path=$CKPT_PATH \
            --eval_dir=$EVAL_DIR\
<<<<<<< HEAD
            --dataset_split_name=test \
=======
            --dataset_split_name=$SPLIT \
>>>>>>> anchor4810
            --model_name=$MODEL_NAME
    ;;
    test)
        EVAL_DIR=$HOME/temp_nfs/ssd_results/
        python test_ssd_network.py \
            --checkpoint_path=$CKPT_PATH \
            --eval_dir=$EVAL_DIR\
            --dataset_split_name=test \
            --model_name=$MODEL_NAME \
            --keep_top_k=20 \
            --keep_threshold=0.05
    ;;
esac


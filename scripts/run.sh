set -x
set -e
export CUDA_VISIBLE_DEVICES=$1
ACTION=$2

SIZE=512
TRAIN_DIR=$HOME/models/ssd-tf/debug
EVAL_DIR=${TRAIN_DIR}/eval/$SPLIT
MODEL_NAME=ssd_${SIZE}_vgg
MIN_OBJECT_COVERED=0.25
MATCH_THRESHOLD=0.25
LOSS_ALPHA=1
LR=0.00001
if [ $ACTION == 'pretrain' ] || [ $ACTION == 'train' ]
then
    IMG_PER_GPU=$3
    
    # get the number of gpus
    OLD_IFS="$IFS" 
    IFS="," 
    gpus=($CUDA_VISIBLE_DEVICES) 
    IFS="$OLD_IFS"
    num_gpus=${#gpus[@]}

    # batch_size = num_gpus * IMG_PER_GPU
    BATCH_SIZE=`expr $num_gpus \* $IMG_PER_GPU`
else
    SPLIT=$3
    EVAL_DIR=${TRAIN_DIR}/eval
    CKPT_PATH=$TRAIN_DIR
fi


case $ACTION in 
    pretrain)
        
        DATASET=$HOME/dataset/SSD-tf/SynthText
        python train_ssd_network.py \
            --dataset_dir=$DATASET \
            --negative_ratio=3 \
            --match_threshold=${MATCH_THRESHOLD} \
            --train_dir=$TRAIN_DIR \
            --learning_rate_decay_type=fixed \
            --learning_rate=0.0001 \
            --dataset_name=synthtext \ #icdar2013 \
            --dataset_split_name=train \
            --model_name=$MODEL_NAME \
            --batch_size=$BATCH_SIZE \
            --should_trace=0 \
            --min_object_covered=${MIN_OBJECT_COVERED} \
            --loss_alpha=${LOSS_ALPHA} \
            --learning_rate=${LR} \
            --max_number_of_steps=100000
    ;;
    train)
        DATASET=$HOME/dataset/SSD-tf/ICDAR
        python train_ssd_network.py \
            --checkpoint_path=/root/models/ssd-pretrain \
            --dataset_dir=$DATASET \
            --negative_ratio=3 \
            --match_threshold=${MATCH_THRESHOLD} \
            --train_dir=$TRAIN_DIR \
            --learning_rate_decay_type=fixed \
            --dataset_name=icdar2013 \
            --dataset_split_name=train \
            --model_name=$MODEL_NAME \
            --batch_size=$BATCH_SIZE \
            --min_object_covered=${MIN_OBJECT_COVERED} \
            --learning_rate=${LR} \
            --loss_alpha=${LOSS_ALPHA} \
            --max_number_of_steps=300000
    ;;
    eval)
        #TRAIN_DIR=$HOME/temp/ssd-text-$SIZE/SynthText-pretrain-cnt/origin-config
        CKPT_PATH=$TRAIN_DIR
        MODEL_NAME=ssd_${SIZE}_vgg
        EVAL_DIR=${TRAIN_DIR}/eval/$SPLIT
       # CUDA_VISIBLE_DEVICES=
        DATASET=$HOME/dataset/SSD-tf/ICDAR
        python eval_ssd_network.py \
            --dataset_dir=$DATASET \
            --checkpoint_path=$CKPT_PATH \
            --eval_dir=$EVAL_DIR\
            --loss_weighted_blocks=${WEIGHTED_BLOCK} \
            --dataset_split_name=$SPLIT \
            --model_name=$MODEL_NAME
    ;;
    test)
        #EVAL_DIR=$HOME/temp_nfs/ssd_results/
	    #CUDA_VISIBLE_DEVICES=
        CKPT_PATH=$4
        if [ $5 ]
        then
            wait_for_checkpoints=$5
        else
            wait_for_checkpoints=0
        fi
        
        python test_ssd_network.py \
            --checkpoint_path=$CKPT_PATH \
            --dataset_split_name=$SPLIT \
            --model_name=$MODEL_NAME \
            --keep_top_k=1000 \
            --wait_for_checkpoints=${wait_for_checkpoints} \
            --keep_threshold=0.7 \
            --nms_threshold=0.45
    ;;
esac


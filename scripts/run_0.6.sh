set -x
set -e
export CUDA_VISIBLE_DEVICES=$1
ACTION=$2

SIZE=512
HOME=/home/dengdan
TRAIN_DIR=$HOME/temp/ssd-text-$SIZE/optm1_0.6
EVAL_DIR=${TRAIN_DIR}/eval/$SPLIT
MODEL_NAME=ssd_${SIZE}_vgg

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
            --match_threshold=0.6 \
            --train_dir=$TRAIN_DIR \
            --learning_rate_decay_type=fixed \
            --learning_rate=0.0001 \
            --dataset_name=synthtext \ #icdar2013 \
            --dataset_split_name=train \
            --model_name=$MODEL_NAME \
            --batch_size=$BATCH_SIZE \
            --should_trace=0 \
#            --gpu_memory_fraction=0.5 \
            --min_object_covered=0.95 \
            --min_width_covered=0.25 \
            --min_height_covered=0.9 \
            --max_number_of_steps=60000
    ;;
    train)
        DATASET=$HOME/dataset/SSD-tf/ICDAR
        python train_ssd_network.py \
            --dataset_dir=$DATASET \
            --negative_ratio=3 \
            --match_threshold=0.6 \
            --train_dir=$TRAIN_DIR \
            --learning_rate_decay_type=fixed \
            --learning_rate=0.0001 \
            --dataset_name=icdar2013 \
            --dataset_split_name=train \
            --model_name=$MODEL_NAME \
            --batch_size=$BATCH_SIZE \
            --should_trace=0 \
            --min_object_covered=0.95 \
#            --gpu_memory_fraction=.5 \
            --min_width_covered=0.3 \
            --min_height_covered=0.95 \
            --max_number_of_steps=400000
    ;;
    eval)
        #TRAIN_DIR=$HOME/temp/ssd-text-$SIZE/SynthText-pretrain-cnt/origin-config
        CKPT_PATH=$TRAIN_DIR
        MODEL_NAME=ssd_${SIZE}_vgg
        EVAL_DIR=${TRAIN_DIR}/eval/$SPLIT
        CUDA_VISIBLE_DEVICES=
        DATASET=$HOME/dataset/SSD-tf/ICDAR
        python eval_ssd_network.py \
            --dataset_dir=$DATASET \
            --checkpoint_path=$CKPT_PATH \
            --eval_dir=$EVAL_DIR\
            --dataset_split_name=$SPLIT \
            --model_name=$MODEL_NAME
    ;;
    test)
        #EVAL_DIR=$HOME/temp_nfs/ssd_results/
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
            --keep_top_k=20 \
            --wait_for_checkpoints=${wait_for_checkpoints} \
            --keep_threshold=0.6
    ;;
esac

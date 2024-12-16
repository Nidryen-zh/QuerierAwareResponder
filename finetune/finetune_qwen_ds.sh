#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_ENDPOINT=https://hf-mirror.com
# export HF_ENDPOINT=https://huggingface.co/
DIR=`pwd`
# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training).
# Please set the options below according to the comments.
# For multi-gpu workers training, these options should be manually set for each worker.
# After setting the options, please run the script on each worker.
# Number of GPUs per GPU worker
GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')
# Number of GPU workers, for single-worker training, please set to 1
NNODES=${NNODES:-1}
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
NODE_RANK=${NODE_RANK:-0}
# The ip address of the rank-0 worker, for single-worker training, please set to localhost
MASTER_ADDR=${MASTER_ADDR:-localhost}
# The port for communication
MASTER_PORT=${MASTER_PORT:-6001}
# Set the path if you do not want to load from huggingface directly
######################################################################################
#                                     profile                                        #
######################################################################################
# PROFILE is defined in finetune_lora_ds_pipline.sh as environment var
source $PROFILE
DS_CONFIG_PATH="finetune/ds_config_zero2.json"
function usage() {
    echo '
Usage: bash finetune/finetune_lora_ds.sh [-m MODEL_PATH] [-d DATA_PATH] [--deepspeed DS_CONFIG_PATH]
'
}
while [[ "$1" != "" ]]; do
    case $1 in
        -m | --model )
            shift
            MODEL=$1
            ;;
        -d | --data )
            shift
            DATA=$1
            ;;
        --deepspeed )
            shift
            DS_CONFIG_PATH=$1
            ;;
        -h | --help )
            usage
            exit 0
            ;;
        * )
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"


torchrun $DISTRIBUTED_ARGS finetune_qwen.py \
    --model_name_or_path $MODEL \
    --data_path ${DATA[@]} \
    --eval_data_path ${DATA_EVAL[@]} \
    --data_root $DATA_ROOT \
    --fp16 True \
    --output_dir "${OUTPUT_PATH}_lr${LR}" \
    --eval_output_dir "${OUTPUT_PATH}_lr${LR}" \
    --logging_dir "${OUTPUT_PATH}_lr${LR}" \
    --num_train_epochs $TRAIN_EPOCH \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_steps $SAVE_STEP \
    --eval_steps 10 \
    --save_total_limit 100 \
    --learning_rate $LR \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length $MAX_LEN \
    --lazy_preprocess True \
    --use_lora True \
    --gradient_checkpointing \
    --deepspeed ${DS_CONFIG_PATH} \
    --multirole_conv True \
    --is_chat_version True 


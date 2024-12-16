#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
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
source $PROFILE
function usage() {
    echo '
Usage: bash finetune/finetune_ds.sh [-m MODEL_PATH] [-d DATA_PATH]
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
    --output_dir "${OUTPUT_PATH}_lr${LR}" \
    --fp16 True \
    --eval_output_dir "${OUTPUT_PATH}_lr${LR}" \
    --logging_dir "${OUTPUT_PATH}_lr${LR}" \
    --num_train_epochs $TRAIN_EPOCH \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_steps 10 \
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
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --use_lora True \
    --deepspeed finetune/ds_config_zero2.json \
    --multirole_conv True \
    --is_chat_version True \
    --role_decomposition True \
    --need_role_pos_ids True \
    --need_role_emb $NEED_ROLE_EMB \
    --only_last_layer_loss $ONLY_LAST_LAYER_LOSS \
    --only_middle_layer_loss $ONLY_MIDDLE_LAYER_LOSS \
    --use_role_cls $USE_ROLE_CLS \
    --no_which_loss $NO_WHICH_LOSS \
    --no_general_role_emb $NO_GENERAL_ROLE_EMB \
    --init_specific_para $INIT_SPECIFIC_PARA \
    --full_para $FULL_PARA \
    --only_push $ONLY_PUSH \
    --weighted_decomposition $WEIGHTED_DECOMPOSITION \
    --np_cls $NP_CLS \
    --use_cluster_specific_mlp $USE_CLUSTER_SPECIFIC_MLP \
    --use_cluster_specific_role_proj $USE_CLUSTER_SPECIFIC_ROLE_PROJ \
    --use_role_proj $USE_ROLE_PROJ \
    --role_proj_dim $ROLE_PROJ_DIM \
    --tuned_norm $TUNED_NORM \
    --average_role_token_as_emb $AVERAGE_ROLE_TOKEN_AS_EMB \
    --lam $LAM \
    --pretrain_role_proj_step $PRETRAIN_ROLE_PROJ_STEP \
    --T $T \
    --K $K \
    --cluster_num $CLUSTER_NUM 

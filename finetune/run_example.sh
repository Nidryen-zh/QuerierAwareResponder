print_cmd()
{
    source $1
    echo "PROFILE: $1"
    echo "------------------------------------------------- runing cmd -------------------------------------------------"
    echo "torchrun $DISTRIBUTED_ARGS finetune.py 
            --model_name_or_path $MODEL 
            --data_path ${DATA[@]}
            --eval_data_path ${DATA_EVAL[@]}
            --data_root $DATA_ROOT
            --learning_rate $LR
            --output_dir $OUTPUT_PATH
            --eval_output_dir $OUTPUT_PATH
            --logging_dir $OUTPUT_LOG_PATH
            --num_train_epochs $TRAIN_EPOCH
            --per_device_train_batch_size $BATCH_SIZE
            --per_device_eval_batch_size `expr $BATCH_SIZE + $BATCH_SIZE`
            --save_steps $SAVE_STEP
            --eval_steps $EVAL_STEP
            --model_max_length $MAX_LEN
            "
    echo "--------------------------------------------------------------------------------------------------------------"
    echo ""
}

## To run the file
# bash finetune/run_example.sh 2>&1 | tee output/run_example_log.txt


############ Qwen ############
for dataset in "swordsman_tongxiangyu" "swordsman_baizhantang" "swordsman_lvxiucai" "palace_zhenhuan" "palace_emperor" "wechat"
do
    export LR=2e-4

    export PROFILE="finetune/configs/$dataset/finetune_qar.sh"
    print_cmd $PROFILE
    bash finetune/finetune_qwen_qar_ds.sh

    export PROFILE="finetune/configs/$dataset/finetune.sh"
    print_cmd $PROFILE
    bash finetune/finetune_qwen_ds.sh
done
############ Qwen ############

############ Llama ############
for dataset in "bigbang_sheldon" "bigbang_leonard" "friends_rachel" "friends_ross" "modernfamily_claire" "modernfamily_phil" 
do
    export LR=2e-4

    export PROFILE="finetune/configs/$dataset/finetune_qar.sh"
    print_cmd $PROFILE
    bash finetune/finetune_llama_qar_ds.sh

    export PROFILE="finetune/configs/$dataset/finetune.sh"
    print_cmd $PROFILE
    bash finetune/finetune_llama_ds.sh
done
############ Llama ############

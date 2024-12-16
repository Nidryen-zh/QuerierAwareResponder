####################################################################
## wechat 
####################################################################
OUTPUT_PATH="output/wechat/finetune"
OUTPUT_LOG_PATH="$OUTPUT_PATH/log"
ROLE_SYSTEM_PROFILE=none
TRAIN_EPOCH=20
SAVE_STEP=1700
EVAL_STEP=10
BATCH_SIZE=4
MAX_LEN=592
TUNED_NORM=False

MODEL="Qwen/Qwen1.5-7B-Chat"
DATA_ROOT="data/wechat/diags_two_role"
DATA_EVAL=()
DATA=()
ROOT="data/wechat/diags_two_role/*"
for dir in $ROOT
do
  if test -d $dir
  then
    for file in ${dir}/*
    do
      if [[ $file =~ 'response_L512_dev.json' ]]; then
        DATA_EVAL[${#DATA_EVAL[@]}]=$file
      fi
      if [[ $file =~ 'response_L512_train.json' ]]; then
        DATA[${#DATA[@]}]=$file
      fi
    done
  fi
done


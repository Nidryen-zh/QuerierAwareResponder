####################################################################
## modernfamily
####################################################################
OUTPUT_PATH="output/modernfamily_phil/finetune"
OUTPUT_LOG_PATH="$OUTPUT_PATH/log"
ROLE_SYSTEM_PROFILE=none
TRAIN_EPOCH=40
SAVE_STEP=1700
EVAL_STEP=25
BATCH_SIZE=4
MAX_LEN=592
TUNED_NORM=False

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
DATA_ROOT="data/modernfamily/diags_two_role_phil"
DATA_EVAL=()
DATA=()
ROOT="data/modernfamily/diags_two_role_phil/*"
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


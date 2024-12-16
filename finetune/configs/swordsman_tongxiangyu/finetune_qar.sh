####################################################################
## my own swordsman
####################################################################
OUTPUT_PATH="output/swordsman_tongxiangyu/finetune_qar"
OUTPUT_LOG_PATH="$OUTPUT_PATH/log"
ROLE_SYSTEM_PROFILE=none
TRAIN_EPOCH=20
SAVE_STEP=1700
EVAL_STEP=25
BATCH_SIZE=4
MAX_LEN=592
NEED_ROLE_EMB=False
ONLY_MIDDLE_LAYER_LOSS=True
ONLY_LAST_LAYER_LOSS=True
USE_ROLE_CLS=True
NP_CLS=True
NO_WHICH_LOSS="general"
NO_GENERAL_ROLE_EMB=True
INIT_SPECIFIC_PARA=True
FULL_PARA=False
ONLY_PUSH=False
WEIGHTED_DECOMPOSITION=False
USE_CLUSTER_SPECIFIC_MLP=False
USE_CLUSTER_SPECIFIC_ROLE_PROJ=True
USE_ROLE_PROJ=True
ROLE_PROJ_DIM=512
PRETRAIN_ROLE_PROJ_STEP=0
LAM=0.5
T=0.1
K=512
CLUSTER_NUM=10
TUNED_NORM=False
MODEL="Qwen/Qwen1.5-7B-Chat"
DATA_ROOT="data/swordsman/diags_two_role_tongxiangyu_clustered"
DATA_EVAL=()
DATA=()
ROOT="data/swordsman/diags_two_role_tongxiangyu_clustered/*"
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


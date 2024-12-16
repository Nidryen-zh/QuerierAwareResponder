# Questioner-Aware Responder

We, for the first time, study questioner-aware LLM personalization to generate customized responses to the same query from different users.

> this version of code contains `finetune` and `finetune_qar` (our design) 

## Files or Folders
- `finetune_llama.py`: training code for Llama, the default setting is lora tuning.
- `finetune_qwen.py`: training code for Qwen, the default setting is lora tuning.
- `trainer.py`: forked from transformers==4.37 with some modifications. Trainer of both Qwen and Llama. 
- `trainer_with_eval.py`: rewrite evaluation_loop function of trainer, adding evaluation with ROUGE.
- `utils.py`: some tool functions.
- `models/modeling_llama.py`: forked from transformers==4.37. Our model `LlamaForCausalLMRoleDecomposition`, `LlamaModelRoleDecomposition`, `LlamaDecoderLayerRoleDecomposition`, `LlamaMLPRoleDecompositionLowRank` are added for Questioner-Aware Personalization for English datasets. 
- `models/modeling_qwen.py`: forked from transformers==4.37. Our model `Qwen2ForCausalLMRoleDecomposition`, `Qwen2ModelRoleDecomposition`, `Qwen2DecoderLayerRoleDecomposition`, `Qwen2MLPRoleDecompositionLowRank` are added for Questioner-Aware Personalization for Chinese datasets. 
- `data`: put the datasets here (Multi-Questioner Dialogue Dataset).
- `output`: we save the results here. 
- `finetune`: config files for different datasets are put here. 

## Install the environment
We provide a `QAR.yaml` file which includes all setting of our running environment. The python version is 3.9.18.

Set your conda path and install the environment before training. 
```sh
conda env create -f QAR.yaml
```
or Insall the necessary package with pip. 

## Dataset
We construct Multi-Questioner Dialogue Dataset (MQDialog) to evaluate questioner-aware personalization. The dataset can be obtained on [huggingface](https://huggingface.co/datasets/Nidhogg-zh/Multi-Questioner_Dialogue). 

## How to run
We provide an example to run our code in `finetune/run_example.sh`.   
We can run the bash file with logging the output.
```sh
bash finetune/run_example.sh 2>&1 | tee output/run_example_log.txt
``` 

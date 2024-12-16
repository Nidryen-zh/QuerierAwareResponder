# OUTPUT & EVALUATION

- Results are saved in `output` folder by default. 
- To evaluate BLEU score, we have `get_evaluation_result_bleu.py` file. Please set your output path in it (i.e. YOUR OUTPUT DIR) and note the language of the dataset. 
```sh
cd output
python get_evaluation_result_bleu.py
```
- To evaluate ROUGE score, we have `get_evaluation_result_rouge.py` file. Please set your output path in it (i.e. YOUR OUTPUT DIR) and note the language of the dataset. 
```sh
cd output
python get_evaluation_result_rouge.py
```
- To evaluate Winning Rate, please prompt GPT-4 to get the results. The detailed prompts are listed in the paper. 
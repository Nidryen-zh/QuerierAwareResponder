from rouge import Rouge
import json
from transformers.trainer_utils import EvalPrediction
import os

def compute_rouge(eval_prediction: EvalPrediction, inputs, output_path: str = None, language="english"):
    if output_path is not None:
        os.makedirs(os.path.split(output_path)[0], exist_ok=True)
    
    # TODO: set language in trainer
    if "swordsman" in output_path or "palace" in output_path or "wechat" in output_path:
        language = "chinese"

    predictions: list[str] = eval_prediction.predictions
    labels: list[str] = eval_prediction.label_ids
    num_samples = len(predictions)
    
    rouge = Rouge()
    rouge_1, rouge_2, rouge_l_f1, rouge_l_p, rouge_l_r = 0, 0, 0, 0, 0
    pred_data = []
    for i in range(num_samples):
        output_text = predictions[i]
        label_text = labels[i]
        input_text = inputs[i]
        pred_data.append({"input": input_text, "pred": output_text, "label": label_text})
        try:
            ## English
            if language == "english":
                result = rouge.get_scores(output_text, label_text, avg=True) 
            ## Chinese
            else:
                result = rouge.get_scores(" ".join(list(output_text)), " ".join(list(label_text)), avg=True)  
            rouge_1 += result['rouge-1']['f']
            rouge_2 += result['rouge-2']['f']
            rouge_l_f1 += result['rouge-l']['f']
            rouge_l_p += result['rouge-l']['p']
            rouge_l_r += result['rouge-l']['r']
        except Exception as e:
            print(f"[During Evaluating Rouge] {e}")
    print('\nAVG Rouge_1: {}, Rouge_2: {}, Rouge_l_f1: {}, Rouge_l_p: {}, Rouge_l_r: {}'.format(rouge_1 / num_samples,
                                                                                              rouge_2 / num_samples,
                                                                                              rouge_l_f1 / num_samples,
                                                                                              rouge_l_p / num_samples,
                                                                                              rouge_l_r / num_samples))
    eval_results = {'Rouge_1': rouge_1 / num_samples,
                    'Rouge_2': rouge_2 / num_samples,
                    'Rouge_l_f1': rouge_l_f1 / num_samples,
                    'Rouge_l_p': rouge_l_p / num_samples,
                    'Rouge_l_r': rouge_l_r / num_samples
                    }
    if output_path is not None:
        json.dump({"pred_results": pred_data, "eval_results": eval_results}, open(output_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    return eval_results

import json
import os
from rouge import Rouge
from transformers.trainer_utils import EvalPrediction


def remove_stopwords(text):
    return text


def compute_rouge(eval_prediction: EvalPrediction, inputs, output_path: str = None, language="english"):
    if output_path is not None:
        os.makedirs(os.path.split(output_path)[0], exist_ok=True)

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
            output_text = remove_stopwords(output_text)
            label_text = remove_stopwords(label_text)

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

    eval_results = {'Rouge_1': rouge_1 / num_samples,
                    'Rouge_2': rouge_2 / num_samples,
                    'Rouge_l_f1': rouge_l_f1 / num_samples,
                    'Rouge_l_p': rouge_l_p / num_samples,
                    'Rouge_l_r': rouge_l_r / num_samples
                    }
    if output_path is not None:
        json.dump({"pred_results": pred_data, "eval_results": eval_results}, open(output_path, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=4)
    return eval_results


task_types = ["YOUR OUTPUT DIR"]

evaluation_results = {}
for task_type in task_types:
    root = f"{task_type}/evaluate_result"
    output_root = f"{task_type}/evaluate_result_total"

    if not os.path.exists(root):
        continue
    print()
    print(task_type)

    best_evaluate_result = {
        "Rouge_1": 0.0,
        "Rouge_2": 0.0,
        "Rouge_l_f1": 0.0,
        "Rouge_l_p": 0.0,
        "Rouge_l_r": 0.0,
        "step": 0,
    }

    dir_list = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    dir_list = sorted(dir_list, key=lambda k: int(k.replace('result-step', '')))

    logs = ""
    for dirc in dir_list:
        if dirc.startswith("result-step"):
            step = int(dirc.replace("result-step", ""))
            d = os.path.join(root, dirc)
            pred_length = 0
            all_inputs, all_preds, all_labels = [], [], []
            for file in os.listdir(d):
                with open(os.path.join(d, file)) as f:
                    pred_result = json.load(f)
                    all_inputs += [item["input"] for item in pred_result['pred_results']]
                    all_preds += [item["pred"] for item in pred_result['pred_results']]
                    all_labels += [item["label"] for item in pred_result['pred_results']]
            metrics = compute_rouge(EvalPrediction(predictions=all_preds, label_ids=all_labels), all_inputs, os.path.join(output_root, dirc, f"pred_result.json"), language="english")
            metrics.update({"step": step})
            if metrics["Rouge_l_f1"] > best_evaluate_result['Rouge_l_f1']:
                best_evaluate_result = metrics
            logs += json.dumps(metrics, ensure_ascii=False) + "\n"
    print(best_evaluate_result)
    evaluation_results[task_type] = best_evaluate_result
    with open(os.path.join(output_root, f"evaluate_result.txt"), 'w') as f:
        f.write(logs)
        f.write("best result:\n")
        f.write(json.dumps(best_evaluate_result, ensure_ascii=False) + "\n")

print("\n" + "-" * 50)
for t in evaluation_results:
    print(t)
    print(evaluation_results[t])
print("-" * 50)
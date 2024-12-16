import json
import os
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import jieba
from transformers.trainer_utils import EvalPrediction


def compute_bleu(eval_prediction: EvalPrediction, inputs, output_path: str = None, language="english"):
    if output_path is not None:
        os.makedirs(os.path.split(output_path)[0], exist_ok=True)

    predictions: list[str] = eval_prediction.predictions
    labels: list[str] = eval_prediction.label_ids
    num_samples = len(predictions)

    stopwords_path = "../evaluate/hit_stopwords.txt"
    with open(stopwords_path) as f:
        stopwords = [line.strip() for line in f.readlines()]

    def word_cut(input_text, stopwords=stopwords):
        ## English
        if language == "english":   
            cuts = input_text.split(" ")
        ## Chinese
        else:
            cuts = jieba.lcut(input_text)
        result = []
        for cut in cuts:
            if cut not in stopwords:
                result.append(cut)
        return result

    candidate, reference = [], []
    for candidate_text, reference_text in zip(predictions, labels):
        candidate.append(word_cut(candidate_text))
        reference.append([word_cut(reference_text)])

    bleu_score_1 = corpus_bleu(reference, candidate, weights=(1,0,0,0))
    bleu_score_2 = corpus_bleu(reference, candidate, weights=(0.5,0.5,0,0))
    bleu_score_3 = corpus_bleu(reference, candidate, weights=(0.33,0.33,0.33,0))
    bleu_score_4 = corpus_bleu(reference, candidate, weights=(0.25,0.25,0.25,0.25))

    eval_results = {'BLEU-1': bleu_score_1,
                    'BLEU-2': bleu_score_2,
                    'BLEU-3': bleu_score_3,
                    'BLEU-4': bleu_score_4,
                    }

    return eval_results

task_types = ["YOUR OUTPUT DIR"]

for task_type in task_types:
    root = f"{task_type}/evaluate_result"
    output_root = f"{task_type}/evaluate_result_total"

    if not os.path.exists(root):
        continue
    print()
    print(task_type)

    best_evaluate_result = {
        'BLEU-1': 0.0,
        'BLEU-2': 0.0,
        'BLEU-3': 0.0,
        'BLEU-4': 0.0,
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
            metrics = compute_bleu(EvalPrediction(predictions=all_preds, label_ids=all_labels), all_inputs, output_path=None, language="english")
            # print(metrics)

            metrics.update({"step": step})
            if metrics["BLEU-1"] > best_evaluate_result['BLEU-1']:
                best_evaluate_result = metrics
            logs += json.dumps(metrics, ensure_ascii=False) + "\n"

    print("best result: ", best_evaluate_result)

    with open(os.path.join(output_root, f"evaluate_result_bleu.txt"), 'w') as f:
        f.write(logs)
        f.write("best result:\n")
        f.write(json.dumps(best_evaluate_result, ensure_ascii=False) + "\n")

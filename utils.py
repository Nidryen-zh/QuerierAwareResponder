import torch
from transformers import AutoModelForCausalLM
from accelerate import dispatch_model


def _device_map(num_gpus, num_layers):
    per_gpu_layers = (num_layers + 2) / num_gpus

    device_map = {
        'transformer.wte': 0,
        'transformer.ln_f': 0,
        'lm_head': num_gpus-1
    }

    used = 1
    gpu_target = 0
    for i in range(num_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0 if gpu_target < num_gpus-1 else 1
        assert gpu_target < num_gpus
        device_map[f'transformer.h.{i}'] = gpu_target
        used += 1

    return device_map


def load_model_on_gpus(model_name_or_path, num_gpus: int = 2):
    num_devices = torch.cuda.device_count()

    if num_gpus == 1:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto',
                                                     trust_remote_code=True).eval()
    elif 1 < num_gpus <= num_devices:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='cpu',
                                                     trust_remote_code=True).eval()
        num_layers = model.config.num_hidden_layers
        device_map = _device_map(num_gpus, num_layers)
        print(device_map)
        model = dispatch_model(model, device_map=device_map)
    else:
        raise KeyError

    return model

# newly defined data collector
from transformers.data.data_collator import DataCollatorWithPadding, pad_without_fast_tokenizer_warning
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from dataclasses import dataclass

@dataclass
class DataCollatorWithPaddingAndPreserveString(DataCollatorWithPadding):

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        standard_columns = ['input_ids', 'attention_mask', 'labels', 'label', 'label_ids']
        other_columns = [key for key in features[0].keys() if key not in standard_columns]
        standard_features = [{k: v for k, v in feature.items() if k in standard_columns} for feature in features]
        other_features = [{k: v for k, v in feature.items() if k in other_columns} for feature in features]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            standard_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        for key in other_columns:
            batch[key] = [feature[key] for feature in other_features]
        return batch


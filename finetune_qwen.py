# This code is based on the revised code from github.com/QwenLM/Qwen

from dataclasses import dataclass, field
import json
import math
import random
import logging
import os
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import BitsAndBytesConfig, deepspeed
from trainer_with_eval import TrainerWithEval

from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType
from evaluate import *
from models.modeling_qwen import Qwen2ForCausalLM, Qwen2ForCausalLMRoleDecomposition

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
SYSTEM = None

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")


@dataclass
class DataArguments:
    # data_path: str = field(
    #     default=None, metadata={"help": "Path to the training data."}
    # )
    data_path: List[str] = field(
        default_factory=lambda: []
    )
    eval_data_path: List[str] = field(
        default_factory=lambda: []
    )
    lazy_preprocess: bool = False
    replay_path: str = field(
        default=None, metadata={"help": "Path to the replay data."}
    )
    is_pretrain: bool = False
    # newly added
    multirole_conv: bool = False
    role_system_profile: str = field(
        default=None, metadata={"help": "Path to the role profile data."}
    )
    target_role: str = field(
        default=None, metadata={"help": "target role of response, None for all roles"}
    )
    need_role_pos_ids: bool = False
    data_root: str = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    use_lwf: bool = False
    eval_output_dir: Optional[str] = field(default=None)
    # newly added
    is_chat_version: bool = False
    role_decomposition: bool = False
    only_roleID: bool = False
    need_role_emb: bool = False
    only_last_layer_loss: bool = False
    use_role_cls: bool = False
    decomposition_pretrain: bool = False
    no_which_loss: str = field(default="none")
    no_general_role_emb: bool = False
    init_specific_para: bool = False
    full_para: bool = False
    only_middle_layer_loss: bool = False
    only_push: bool = False
    weighted_decomposition: bool = False
    np_cls: bool = False
    use_role_proj: bool = False
    use_cluster_specific_mlp: bool = False
    use_cluster_specific_role_proj: bool = False
    role_proj_dim: int = 512
    tuned_norm: bool = False
    average_role_token_as_emb: bool = True
    lam: float = 0.1
    pretrain_role_proj_step: int = 0
    T: float = 0.1
    K: int = 512
    cluster_num: int = 10


@dataclass
class LoraArguments:
    lora_r: int = 16
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj",]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        # if trainer.args.use_lora:
        #     state_dict = get_peft_state_maybe_zero_3(
        #         trainer.model.named_parameters(), bias
        #     )
        # else:
        #     state_dict = trainer.model.state_dict()

        ### Save all the parameters
        state_dict = trainer.model.state_dict()
        
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)

def preprocess_multirole_eval(sources, target_role, tokenizer, max_len, system_message: str = SYSTEM, role_pair_id: int = None):
    im_start = tokenizer("<|im_start|>").input_ids
    im_end = tokenizer("<|im_end|>").input_ids
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _target_role = im_start + tokenizer(target_role).input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    pos_ids = []

    for i, source in enumerate(sources):
        assert source[-1]['from'] == target_role, "[ZH ERROR] last sentence is not from {target}".format(target=target_role)
        input_id = []
        pos_id = []
        target = ""
        if system_message is not None and len(system_message) > 0:
            system = im_start + _system + tokenizer(system_message).input_ids + im_end + nl_tokens
            input_id += system
            pos_id += [0] * len(input_id)
        assert len(pos_id) == len(input_id), "len input_id != pos_id"
        for j, sentence in enumerate(source):
            role = sentence["from"]
            if role == target_role and j == len(source) - 1:
                contain_target_role = True
                target = sentence['value']
                _input_id = im_start + tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = im_start + tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + im_end + nl_tokens
            if role == target_role:
                if j == len(source) - 1:
                    pos_id += [0] + [1] * (len(_input_id) - 1)
                else:
                    pos_id += [0] + [1] * (len(_input_id) - 3) + [0] + [0]
            else:
                pos_id += [0] + [role_pair_id] * (len(_input_id) - 3) + [0] + [0]
            input_id += _input_id
        assert contain_target_role, "[ZH ERROR] There are no target role or the last conversation is not from target role"
        input_id = [tokenizer.pad_token_id] * (max_len - len(input_id)) + input_id
        input_ids.append(input_id[:max_len])
        pos_id = [0] * (max_len - len(pos_id)) + pos_id
        pos_ids.append(pos_id[:max_len])
        target_id = tokenizer(target).input_ids
        targets.append(target_id + [IGNORE_TOKEN_ID] * (max_len - len(target_id)))
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)
    pos_ids = torch.tensor(pos_ids, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id), # attention mask is no use
        role_pos_ids=pos_ids,
    )


def preprocess_multirole(sources, target_role, tokenizer, max_len, system_message: str = SYSTEM, role_pair_id: int = None):

    im_start = tokenizer("<|im_start|>").input_ids
    im_end = tokenizer("<|im_end|>").input_ids
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _target_role = im_start + tokenizer(target_role).input_ids + nl_tokens

    input_ids, targets = [], []
    pos_ids = []

    for i, source in enumerate(sources):
        assert source[-1]['from'] == target_role, "[ZH ERROR] last sentence is not from {target}\n{source}".format(target=target_role, source=source)
        if source[0]["from"] == target_role:
            source = source[1:]
        input_id, target = [], []
        pos_id = []
        if system_message is not None and len(system_message) > 0:
            system = im_start + _system + tokenizer(system_message).input_ids + im_end + nl_tokens
            input_id += system
            target += im_start + [IGNORE_TOKEN_ID] * (len(system) - 3) + im_end + nl_tokens
            pos_id += [0] * len(input_id)
        assert len(input_id) == len(target), "len input_id != target"
        assert len(pos_id) == len(input_id), "len input_id != pos_id"
        contain_target_role = False
        for j, sentence in enumerate(source):
            role = sentence["from"]
            _input_id = im_start + tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + im_end + nl_tokens
            input_id += _input_id
            if role != target_role:
                _target = im_start + [IGNORE_TOKEN_ID] * (len(_input_id) - 3) + im_end + nl_tokens
            else:
                contain_target_role = True
                _target = im_start + [IGNORE_TOKEN_ID] * (len(tokenizer(role).input_ids) + 1) + _input_id[len(tokenizer(role).input_ids) + 2:-2] + im_end + nl_tokens
            if role == target_role:
                pos_id += [0] + [1] * (len(_input_id) - 3) + [0] + [0]
            else:
                pos_id += [0] + [role_pair_id] * (len(_input_id) - 3) + [0] + [0]
            target += _target
        assert contain_target_role, f"[ZH ERROR] There are no target role or the last conversation is not from target role"
        assert len(input_id) == len(target), "Length of input_id and target is different"
        assert len(pos_id) == len(input_id), "Length of pos_id and input_id is different"
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        pos_id += [0] * (max_len - len(pos_id))
        if len(input_id) > max_len: 
            rank0_print(f"[ZH WARNING] input_id is longer than {max_len}")
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
        pos_ids.append(pos_id[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)
    pos_ids = torch.tensor(pos_ids, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        role_pos_ids=pos_ids,
    )


class LazySupervisedMultiroleDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int, target_role: str = None,
                       is_train: bool = True, role_system_profile: dict = None, need_role_pos_ids: bool = False):
        super(LazySupervisedMultiroleDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.target_role = target_role
        self.is_train = is_train
        self.role_system_profile = role_system_profile if role_system_profile is not None else {}
        self.need_role_pos_ids = need_role_pos_ids

        rank0_print("Formatting inputs...Skip in lazy mode...in multi-role mode")
        self.tokenizer = tokenizer
        self.raw_data = self.segment(raw_data, max_len, target_role)
        rank0_print("Data loaded len: {}".format(len(self.raw_data)))
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        target_role = self.raw_data[i]["conversations"][-1]["from"]

        system_message = self.role_system_profile[target_role] if target_role in self.role_system_profile else None
        ### for few-shot testing
        # input_role = self.raw_data[i]["input_role"]
        # system_message = self.role_system_profile[input_role] if input_role in self.role_system_profile else None

        role_pair_id = int(self.raw_data[i]['role_pair_id']) + 1
        if 'cluster_id' in self.raw_data[i]:
            cluster_id = int(self.raw_data[i]['cluster_id'])

        if self.is_train:
            ret_raw = preprocess_multirole([self.raw_data[i]["conversations"]], target_role, self.tokenizer, self.max_len, system_message, role_pair_id)
        else:
            ret_raw = preprocess_multirole_eval([self.raw_data[i]["conversations"]], target_role, self.tokenizer, self.max_len, system_message, role_pair_id)
        ret = dict(
            input_ids=ret_raw["input_ids"][0],
            labels=ret_raw["labels"][0],
            attention_mask=ret_raw["attention_mask"][0],
        )
        if self.need_role_pos_ids:
            ret.update({"role_pos_ids": ret_raw['role_pos_ids'][0]})
        ret.update({'role_pair_id': role_pair_id})
        if 'cluster_id' in self.raw_data[i]:
            ret.update({'cluster_ids': cluster_id})
        self.cached_data_dict[i] = ret
        return ret

    def segment(self, raw_data, max_len, target_role):
        '''
        TODO: pre-process raw_data with different max len to avoid overflow
        Now: filter conversations do not contain target role
        '''
        if target_role is None:
            return raw_data
        new_data = []
        for item in raw_data:
            if target_role in [r['from'] for r in item['conversations']]:
                new_data.append(item)

        return new_data

def load_json_data(data_path):
    try:
        if type(data_path) == list:
            json_data = []
            for dp in data_path:
                json_data += json.load(open(dp, "r"))
        else:
            json_data = json.load(open(data_path, "r"))
        return json_data
    except:
        return None


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = LazySupervisedMultiroleDataset
    rank0_print("Loading data...")

    # get training json
    rank0_print("[DEBUG] training data path:", data_args.data_path)
    train_json = load_json_data(data_args.data_path)

    role_system_profile = load_json_data(data_args.role_system_profile)
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len, target_role=data_args.target_role, role_system_profile=role_system_profile, need_role_pos_ids=data_args.need_role_pos_ids)

    if data_args.eval_data_path:
        # get dev json
        rank0_print("[DEBUG] evaluating data path:", data_args.eval_data_path)
        eval_json = load_json_data(data_args.eval_data_path)
        role_system_profile = load_json_data(data_args.role_system_profile)
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len, target_role=data_args.target_role, is_train=False, role_system_profile=role_system_profile, need_role_pos_ids=data_args.need_role_pos_ids)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def print_trainable_parameter(model):
    rank0_print("======================================================")
    rank0_print("=                   trainable para                   =")
    rank0_print("======================================================")
    for n, p in model.named_parameters():
        if p.requires_grad == True:
            rank0_print(n)

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # This serves for single-gpu qlora.
    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1))==1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    device_map = "cuda"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are incompatible with QLoRA."
            )

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False
    if training_args.use_lwf:
        config.auto_map['AutoModelForCausalLM'] = "modeling_qwen.QWenLMHeadModelLWF"

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # Load model and tokenizer
    config.pad_token_id = config.eos_token_id
    if training_args.role_decomposition:
        role_pair_id_file = os.path.join(data_args.data_root, "configs/role_pair_id.json")
        role_pair_id_list = json.load(open(role_pair_id_file, "r"))
        role_pair_num = len(list(role_pair_id_list.values())[0]) + 2
        config.update({"role_pair_num": role_pair_num, "role_padding_idx": 0})
        model = Qwen2ForCausalLMRoleDecomposition.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype="auto",
            cache_dir=training_args.cache_dir,
            device_map=device_map,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=False,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
            if training_args.use_lora and lora_args.q_lora
            else None,
            decomposition=not data_args.is_pretrain and not training_args.only_roleID,
            need_role_emb=training_args.need_role_emb,
            only_last_layer_loss=training_args.only_last_layer_loss,
            use_role_cls=training_args.use_role_cls,
            decomposition_pretrain=training_args.decomposition_pretrain,
            no_which_loss=training_args.no_which_loss,
            no_general_role_emb=training_args.no_general_role_emb,
            full_para=training_args.full_para,
            only_middle_layer_loss=training_args.only_middle_layer_loss,
            only_push=training_args.only_push,
            weighted_decomposition=training_args.weighted_decomposition,
            NP_cls=training_args.np_cls,
            use_role_proj=training_args.use_role_proj,
            use_cluster_specific_mlp=training_args.use_cluster_specific_mlp,
            use_cluster_specific_role_proj=training_args.use_cluster_specific_role_proj,
            role_proj_dim=training_args.role_proj_dim,
            lam=training_args.lam,
            average_role_token_as_emb=training_args.average_role_token_as_emb,
            output_dir=training_args.output_dir,
            pretrain_role_proj_step=training_args.pretrain_role_proj_step,
            T=training_args.T,
            K=training_args.K,
            cluster_num=training_args.cluster_num
        )
        if training_args.weighted_decomposition:
            model.init_weighted_decomposition()
    else:
        # model = transformers.AutoModelForCausalLM.from_pretrained(
        model = Qwen2ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype="auto",
            cache_dir=training_args.cache_dir,
            device_map=device_map,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=False,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
            if training_args.use_lora and lora_args.q_lora
            else None,
            output_dir=training_args.output_dir,
        )

    if training_args.fp16:
        model.half()
    if training_args.bf16:
        model.bfloat16()

    if training_args.init_specific_para:
        model.init_specific_para()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("training_args tuned norm:", training_args.tuned_norm)
    if training_args.use_lora:
        if training_args.role_decomposition:
            if not training_args.full_para:
                modules_to_save = ["role_embed_tokens", "gate_proj_low_rank", "up_proj_low_rank", "down_proj_low_rank", "model.model.norm_specific", "model.model.norm_role_proj"]
            else:
                modules_to_save = ["role_embed_tokens", "model.model.norm_specific", "model.model.norm_role_proj"]
            if training_args.use_role_proj:
                if training_args.use_cluster_specific_role_proj:
                    modules_to_save += [f"role_proj.{i}" for i in range(10)] + [f"role_avg_proj.{i}" for i in range(10)]
                else:
                    modules_to_save += ["role_proj", "role_avg_proj"]
            
            if training_args.tuned_norm:
                if training_args.use_cluster_specific_role_proj:
                    modules_to_save += ["model.model.norm"]
                else:
                    modules_to_save += ["model.model.norm"]

        elif (lora_args.q_lora or 'chat' in model_args.model_name_or_path.lower() or training_args.is_chat_version) and training_args.tuned_norm:
            modules_to_save = ["model.model.norm"]
        else:
            modules_to_save = []
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save  # This argument serves for adding new tokens.
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)

        # Print peft trainable params
        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    # print_trainable_parameter(model)

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )

    # Start trainner
    trainer = TrainerWithEval(
        model=model, tokenizer=tokenizer, args=training_args, **data_module, compute_metrics=compute_rouge
    )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)


if __name__ == "__main__":
    train()
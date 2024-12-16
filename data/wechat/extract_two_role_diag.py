import json 
import os 
from pypinyin import pinyin, lazy_pinyin
import random

from clean_diag import *

file_name = "wechat"

def get_concat_diags(root="diags_raw"):
    diags = []
    for d in os.listdir(root):
        for file in os.listdir(os.path.join(root, d)):
            if file.endswith(".json"):
                for item in json.load(open(os.path.join(root, d, file), encoding='utf-8')):
                    diags.append({"id": "episode_{}_chunk_{}".format(item['episode_idx'], item['chunk_idx']), 
                                "conversations": item['diag']})
    print(f"[Whole diags length]: {len(diags)}")
    return diags

def get_role_list(diags):
    role_dict = {}
    for diag in diags:
        for conv in diag['conversations']:
            role = conv['from']
            if role not in role_dict:
                role_dict[role] = 1
            else:
                role_dict[role] += 1
    sorted_role_dict = {}
    for k, v in sorted(role_dict.items(), key=lambda x: (-x[1], x[0])):
        sorted_role_dict[k] = v
    # filter out roles whose conversation is less than 50
    role_list = []
    for role in role_dict:
        if role_dict[role] >= 40:
            role_list.append(role)    
    role_pinyin = {role: "".join(role) for role in role_list}
    return role_pinyin, sorted_role_dict


def extract_conversations_between_two_role(conversations, role1, role2):
    def maybe_append_conversation(conv, role1, role2):
        roles = set([item['from'] for item in conv])
        if role1 in roles and role2 in roles and len(roles) == 2:
            return True
        return False

    new_conversations = []
    start_flag = False
    start_idx = 0
    end_idx = 0
    for i, conversation in enumerate(conversations):
        if not start_flag and (conversation['from'] == role1 or conversation['from'] == role2):
            start_flag = True
            start_idx = i
            end_idx = i
            continue
        if start_flag:
            if conversation['from'] == role1 or conversation['from'] == role2:
                end_idx = i
                continue
            elif maybe_append_conversation(conversations[start_idx:end_idx+1], role1, role2):
                new_conversations.append(conversations[start_idx:end_idx+1])
            start_flag = False
    return new_conversations

def extract_diag_between_two_role(diags, role1, role2):
    new_diags = []
    for item in diags:
        new_conversations = extract_conversations_between_two_role(item['conversations'], role1, role2)
        if len(new_conversations) > 0:
            for i, conversation in enumerate(new_conversations):
                new_diags.append({"id": "{}_index_{}".format(item["id"], i), "conversations": conversation})
    return new_diags

def split_train_and_dev(data, prob=0.8):
    random.shuffle(data)
    index = max(int(len(data)*prob), 1)
    train_data = data[:index]
    dev_data = data[index:]
    return train_data, dev_data

def split_diag(data, max_length=2048):
    new_data = []
    for item in data:
        id = item["id"]
        conversations = item['conversations']
        count_len = [0 for _ in range(len(conversations) + 1)]  # [0, len(1st conv), len(1st + 2nd conv), ...]
        count = 0  # the number of final parts
        conversations = remove_blank_value_for_conv(conversations)
        for i, conv in enumerate(conversations):
            if i == 0:
                count_len[i + 1] = len("{}\n{}".format(conv['from'], conv['value']))
                continue
            count_len[i + 1] = count_len[i] + len("{}\n{}".format(conv['from'], conv['value']))
            flag = False
            for start_id in range(0, i + 1):
                if count_len[i + 1] - count_len[start_id] < max_length:
                    flag = True
                    break
            if flag:
                new_conv = conversations[start_id:i + 1]
                new_id = id + f"_part{count}"
                count += 1
                new_data.append({"id": new_id, "conversations": new_conv})
    return new_data

def extract_diag_for_target_from_role_conv(diags, role_pair_id, target_role, target_role_pinyin, input_role, input_role_pinyin):
    new_diags = []
    for item in diags:
        if item['conversations'][-1]['from'] == target_role:
            new_diags.append(item)
        item['target_role'] = target_role
        item['target_role_short'] = target_role_pinyin
        item['input_role'] = input_role
        item['input_role_short'] = input_role_pinyin
        item['role_pair_id'] = role_pair_id[target_role][input_role]
    return new_diags

def merge_diags(diags):
    new_diags = []
    for item in diags:
        convs = []
        last_user = None
        text = ""
        for conv in item['conversations']:
            user = conv['from']
            if user != last_user and last_user is not None:
                convs.append({"from": last_user, "value": text.strip(), "timestamp": last_timestamp})
                text = ""
            last_user = user
            text += conv["value"] + ' ' 
            last_timestamp = conv['timestamp']
        if len(text) > 0:
            convs.append({"from": last_user, "value": text.strip(), "timestamp": last_timestamp})
        new_diags.append({"id": item['id'], 'conversations': convs})
    return new_diags

if __name__ == "__main__":
    # Step2-1: get filtered role list and its pinyin
    roles_all = [f.replace(".json", "") for f in os.listdir("data") if ".json" in f]
    roles = []
    for role in roles_all:
        data_raw = json.load(open(f"data/{role}.json", encoding='utf-8'))
        convs = []
        for item in data_raw:
            convs += item["conversations"]
        data_len = len(convs)
        print(role, data_len)
        if data_len > 200:
            roles.append(role)
    roles = sorted(roles, key=lambda x: -len(json.load(open(f"data/{x}.json", encoding='utf-8'))))
    # clean data
    role_pinyin = {role: "".join(lazy_pinyin(role)) for role in roles}
    role1 = "<|USER_NAME|>"
    role1_pinyin = "zenghang"
    print(f"[Role list length]: {len(role_pinyin)}")
    os.makedirs("diags_two_role/configs", exist_ok=True)
    json.dump(role_pinyin, open("diags_two_role/configs/role_list.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    # Step2-2 ~ 4
    for role2 in role_pinyin:
        # Step 2-2: extract diag between two role
        role2_pinyin = role_pinyin[role2]
        new_diags = json.load(open(os.path.join("data", f"{role2}.json"), encoding='utf-8'))
        new_diags = merge_diags(new_diags)
        new_diags = clean_diag(new_diags)
        output_dir = f'diags_two_role/{role1_pinyin}_{role2_pinyin}'
        os.makedirs(output_dir, exist_ok=True)
        json.dump(new_diags, open(os.path.join(output_dir, f'{file_name}_diags_{role1_pinyin}_{role2_pinyin}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
        print(f"[Diags between {output_dir}]:", len(new_diags))

        # Step 2-3: split training set and validation set
        train_data, dev_data = split_train_and_dev(new_diags, prob=0.8)
        json.dump(train_data, open(os.path.join(output_dir, f'{file_name}_diags_{role1_pinyin}_{role2_pinyin}_train.json'), "w", encoding="utf-8"), ensure_ascii=False, indent=4)
        json.dump(dev_data, open(os.path.join(output_dir, f'{file_name}_diags_{role1_pinyin}_{role2_pinyin}_dev.json'), "w", encoding="utf-8"), ensure_ascii=False, indent=4)
        print("[Split train and dev] {} - {}: {}".format(role1, role2, len(new_diags)), '->', len(train_data), len(dev_data))

        # Step 2-4: split diags with sliding window
        new_train_data = split_diag(train_data, max_length=512)
        new_train_data = clean_diag_with_repeated(new_train_data)
        new_dev_data = split_diag(dev_data, max_length=512)
        new_dev_data = clean_diag_with_repeated(new_dev_data)
        json.dump(new_train_data, open(os.path.join(output_dir, f"{file_name}_diags_{role1_pinyin}_{role2_pinyin}_L512_train.json"), "w", encoding='utf-8'), ensure_ascii=False, indent=4)
        json.dump(new_dev_data, open(os.path.join(output_dir, f"{file_name}_diags_{role1_pinyin}_{role2_pinyin}_L512_dev.json"), "w", encoding='utf-8'), ensure_ascii=False, indent=4)
        print("[Split diag with sliding window] {}, {}".format(len(new_train_data), len(new_dev_data)))

    # Step 2-5: extract diag for target role
    role_pair_id = {role1: {role2: i+1 for i, role2 in enumerate(role_pinyin)}}
    json.dump(role_pair_id, open("diags_two_role/configs/role_pair_id.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    for role2 in role_pinyin: 
        role2_pinyin = role_pinyin[role2]
        input_dir = f'diags_two_role/{role1_pinyin}_{role2_pinyin}'
        input_file = f"{file_name}_diags_{role1_pinyin}_{role2_pinyin}_L512_train.json"
        output_file = f"{file_name}_diags_{role1_pinyin}_{role2_pinyin}_{role1_pinyin}_response_L512_train.json"
        diags = json.load(open(os.path.join(input_dir, input_file), encoding='utf-8'))
        new_diags = extract_diag_for_target_from_role_conv(diags, role_pair_id, role1, role1_pinyin, role2, role2_pinyin)
        new_diags = clean_diag(new_diags)
        json.dump(new_diags, open(os.path.join(input_dir, output_file), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

        input_file = f"{file_name}_diags_{role1_pinyin}_{role2_pinyin}_L512_dev.json"
        output_file = f"{file_name}_diags_{role1_pinyin}_{role2_pinyin}_{role1_pinyin}_response_L512_dev.json"
        diags = json.load(open(os.path.join(input_dir, input_file), encoding='utf-8'))
        new_diags = extract_diag_for_target_from_role_conv(diags, role_pair_id, role1, role1_pinyin, role2, role2_pinyin)
        new_diags = clean_diag_with_repeated(new_diags)
        json.dump(new_diags, open(os.path.join(input_dir, output_file), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
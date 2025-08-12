import json
from copy import deepcopy

def remove_blank_value_for_conv(conversations):
    new_conversations = deepcopy(conversations)
    for conv in conversations:
        if len(conv['value']) == 0:
            new_conversations.remove(conv)
    return new_conversations

def remove_blank_value(data):
    for item in data:
        item['conversations'] = remove_blank_value_for_conv(item['conversations'])
    return data

def remove_conv_with_one_person(data):
    new_data = deepcopy(data)
    for item in data:
        if len(set([conv['from'] for conv in item['conversations']])) <= 1:
            new_data.remove(item)
    return new_data

def compute_similarity(conv1, conv2):
    conv1_list = [item['from'] + item['value'] for item in conv1]
    conv2_list = [item['from'] + item['value'] for item in conv2]
    if len(conv1_list) < len(conv2_list):
        count = 0
        for conv in conv1_list:
            if conv in conv2_list:
                count += 1
        return count / len(conv1_list)
    else:
        count = 0
        for conv in conv2_list:
            if conv in conv1_list:
                count += 1
        return count / len(conv2_list)


def remove_repeated_value(data):
    visited_item = []
    new_data = deepcopy(data)
    for item1 in data:
        need_remove = []
        item1_deleted = False
        for item2 in visited_item:
            sim = compute_similarity(item1['conversations'], item2['conversations'])
            if sim > 0.8:
                # print("item1", item1, "item2", item2, sep='\n')
                if len(item1['conversations']) < len(item2['conversations']):
                    new_data.remove(item1)
                    item1_deleted = True
                    break
                else:
                    new_data.remove(item2)
                    need_remove.append(item2)
        for item in need_remove:
            visited_item.remove(item)
        if not item1_deleted:
            visited_item.append(item1)
    return new_data

def remove_diags_without_target_role_and_input_role(data):
    new_data = deepcopy(data)
    for item in data:
        roles = [c['from'] for c in item['conversations']]
        if item['input_role'] not in roles or item['target_role'] not in roles:
            new_data.remove(item)
            continue
        last_response_is_input_role = False
        for i in range(len(item['conversations'])-1, -1, -1):
            if item['conversations'][i]['from'] == item['target_role']:
                continue
            else:
                if item['conversations'][i]['from'] == item['input_role']:
                    last_response_is_input_role = True 
                break 
        if not last_response_is_input_role:
            new_data.remove(item)
            print(item)
    return new_data

def re_id_to_avoid_repeat(data):
    new_data = deepcopy(data)
    visited_id = {}
    for i, item in enumerate(data):
        if item['id'] not in visited_id:
            visited_id[item['id']] = 0
        else:
            visited_id[item['id']] += 1
        new_data[i]['id'] += '_piece_{}'.format(visited_id[item['id']])
    return new_data

def clean_diag(diags):
    diags = remove_repeated_value(diags)
    diags = remove_blank_value(diags)
    diags = remove_conv_with_one_person(diags)
    return diags

def clean_diag_with_repeated(diags):
    diags = remove_blank_value(diags)
    diags = remove_conv_with_one_person(diags)
    return diags
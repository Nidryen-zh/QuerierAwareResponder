import json 
import os
import re
from datetime import datetime 

def extract_line(line):
    '''
    flag == 0: empty line
    flag == 1: timestamp and user name
    flag == 2: content  
    '''
    flag = 0
    if line == '\n':
        return (""), flag
    timestamp = re.findall(r"\d+-\d+-\d+ \d+:\d+:\d+", line)
    if len(timestamp) > 0:
        flag = 1
        timestamp = datetime.strptime(timestamp[0], "%Y-%m-%d %H:%M:%S")
        user_name = line.rsplit(" ", 1)[-1]
        return (timestamp, user_name), flag
    else:
        flag = 2
        return (line,), flag
    
def timestamp_distance(time1, time2):
    if time2 is None: # at the beginning of process
        return False
    days = (time1 - time2).days
    if days < 0:
        days = (time2 - time1).days
        seconds = (time2 - time1).seconds
    else:
        days = (time1 - time2).days
        seconds = (time1 - time2).seconds
    seconds += days * 3600 * 24
    if seconds > 3600 * 3:
        return True 
    else:
        return False 


for d in os.listdir("data_raw"):
    target_name = d.split("(")[0]
    with open(os.path.join("data_raw", d, f"{target_name}.txt"), encoding='utf-8') as f:
        lines = f.readlines()

    records = []
    convs = []
    timestamp_last = None
    text = ""
    for line in lines:
        content, flag = extract_line(line)
        if flag == 0:
            conv = {"from": user_name[:-1], "value": text[:-1], "timestamp": str(timestamp)}
            if timestamp_distance(timestamp, timestamp_last):
                text = ""
                records.append({"id": f"conv_{len(records)}_{timestamp_last}", "conversations": convs})
                convs = [conv]
            else:
                convs.append(conv)
                text = ""
            timestamp_last = timestamp
        elif flag == 1:
            timestamp, user_name = content 
        else:
            text += content[0]

    if len(convs) != 0:
        records.append({"id": f"conv_{len(records)}_{timestamp_last}", "conversations": convs})

    os.makedirs("data", exist_ok=True)
    json.dump(records, open(os.path.join("data", f"{target_name}.json"), "w", encoding='utf-8'), ensure_ascii=False, indent=4)

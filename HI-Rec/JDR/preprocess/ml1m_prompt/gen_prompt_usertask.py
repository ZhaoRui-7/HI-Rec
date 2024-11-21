#没有把ID变成占位符
import json
import random
import re
# Age和Occupation的映射
age_mapping = {
    1: "Under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+"
}

occupation_mapping = {
    0: "other or not specified",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer"
}

# 加载user和item的内部ID到外部ID的映射
with open('path to user2id.json', 'r') as f:
    user_id_to_external = json.load(f)

with open('path to item2id.json', 'r') as f:
    item_id_to_external = json.load(f)

# 创建外部ID到内部ID的反向映射
external_to_user_id = {int(v): int(k) for k, v in user_id_to_external.items()}
external_to_item_id = {int(v): int(k) for k, v in item_id_to_external.items()}

# 读取user和item的外部ID映射
user_external_ids = {}
with open('path to ml1m_userid_2_HID.txt', 'r') as f:
    for line in f:
        id1, id3 = line.strip().split(',')
        user_external_ids[int(id1)] = id3

item_external_ids = {}
with open('path to ml1m_itemid_2_HID.txt', 'r') as f:
    for line in f:
        id1, _, id3 = line.strip().split(',')  # 忽略中间的无用信息
        item_external_ids[int(id1)] = id3


def extract_title(label):
    # 匹配格式：Thin Red Line, The (1998) 或 Felicia's Journey (1999)
    pattern = r"^(.*?)(?:, The)? \(\d{4}\)$"

    match = re.match(pattern, label)
    if match:
        # 提取并返回标题
        return match.group(1)
    else:
        return None
random_numbers = {i: (random.randint(0, 49), random.randint(0, 49)) for i in range(4001)}
# 读取train_output.json文件
with open('path to train_output.json', 'r') as f:
    data = [json.loads(line) for line in f]

training_samples = []

for user_record in data:
    user_id = user_record['UserID']

    gender = 'Female' if user_record['gender']=='F' else 'Male'
    age = age_mapping[int(user_record['age'])]
    occupation = occupation_mapping[int(user_record['occupation'])]
    history = user_record['history']


    for i in range(10, min(len(history),30)):\
        if i>=10:
            prompt_history = history[i-10:i]
        else:
            prompt_history = history[:i]
        label = history[i]
        item_info =""
        for movie in prompt_history:
            iid = item_external_ids[external_to_item_id[int(movie['ItemID'])]]
            s = ""
            s+=(f"[i_{iid}]")
            for i in range(len(str(iid))):
                s+=f"[i_{iid[:i+1]}]"
            output_text = f"MovieID: {s}, title: {movie['title']}, type: {movie['type']}, score: {movie['score']};"
            item_info+=output_text

        s = ""
        for i in range(len(str(user_id))):
            s+=(f"[u_{user_id[:i+1]}]")
        prompt = (
            f"You are a professional movie recommendation analyst. Based on the following user profile and the information about movies the user has previously watched, please predict what movie the user might watch next. Let's think step by step, using common sense and knowledge from various disciplines.\\n "
            f"The user's basic profile is as follows: UserID: {user_id},Gender:{gender}, Age:{age}, Occupation:{occupation}.\\n "
            f"The movies the user has previously watched are as follows:{item_info}."
        )
        x = label["type"].replace('|', ', ')
        lid = item_external_ids[external_to_item_id[int(label['ItemID'])]]
        s = ""
        s+=(f"[i_{lid}]")
        for i in range(len(str(lid))):
            s+=(f"[i_{lid[:i+1]}]")
        label = (f'The movie the user might watch next is MovieID: {s}, title: {label["title"]}, type is {x}.')
        # label = ""

        training_sample = {
            "prompt": prompt,
            "label":label,
            "user_id": [user_record['UserID']],
            "item_id": [item["ItemID"] for item in prompt_history]
        }

        training_samples.append(training_sample)

# 将生成的训练样本写入文件
with open('path to train_samples.json', 'w') as f:
    for sample in training_samples:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')


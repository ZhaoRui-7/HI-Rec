import json

# 读取 JSON 文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 读取 .dat 文件
def load_dat(file_path, delimiter='::'):
    data = {}
    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split(delimiter)
            data[parts[0]] = parts[1:]
    return data

# 加载所有文件
sequential_data = load_json('path to sequential_data.json')
item2id = load_json('path to item2id.json')
user2id = load_json('path to user2id.json')
users_dat = load_dat('path to users.dat')
movies_dat = load_dat('path to movies.dat')
# print(user2id)
# 创建 item_id 到电影信息的映射
item_info = {}
for origin_item_id, (title, genres) in movies_dat.items():
    if origin_item_id in item2id:
        item_id = item2id[origin_item_id]
        item_info[item_id] = {"title": title, "genres": genres}
# print(item_info)
# 创建 user_id 到用户信息的映射
user_info = {}
for origin_user_id, (gender, age, occupation, zip_code) in users_dat.items():
    if origin_user_id in user2id:
        user_id = user2id[origin_user_id]
        user_info[user_id] = {"gender": gender, "age": age, "occupation": occupation}
# print(user_info[1])
# 生成每个用户的 JSON 数据
output_data = []
for user_id, (watched_items, ratings,times) in sequential_data.items():
    user_id = int(user_id)
    if user_id in user_info:
        # print("yes")
        user_data = {"UserID": user_id, "gender": user_info[user_id]["gender"], "age": user_info[user_id]["age"], "occupation": user_info[user_id]["occupation"], "history": []}
        max_len = min((len(watched_items) + 1) // 2, 30)
        i = 0
        for item_id, rating,time in zip(watched_items, ratings,times):
            if i<max_len:
                item_id = int(item_id)
                i+=1
                if rating > 3 and item_id in item_info:
                    item_data = {"ItemID": item_id, "title": item_info[item_id]["title"], "type": item_info[item_id]["genres"], "score": rating,"time":time}
                    user_data["history"].append(item_data)
            else:
                break
        output_data.append(user_data)

# 保存结果到 JSON 文件
with open('save to output.json', 'w', encoding='utf-8') as f:
    for user_data in output_data:
        json_line = json.dumps(user_data, ensure_ascii=False)
        f.write(json_line + '\n')

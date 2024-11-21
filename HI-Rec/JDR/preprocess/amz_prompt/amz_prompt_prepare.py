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
datamap = load_json('path to datamaps.json')
sequential_data = load_json('path to sequential_data.json')
movies_dat = load_dat('path to item.dat')
itemattribute = load_json('path to item_info.json')

# 创建 item_id 到电影信息的映射
item_info = {}
for id2, (title, genres) in movies_dat.items():
    if str(id2) in datamap["id2item"]:
        item_id = datamap["id2item"][id2]
        item_info[int(id2)] = itemattribute[item_id]
# print(item_info)
# 创建 user_id 到用户信息的映射
user_info = datamap['id2user']
# 生成每个用户的 JSON 数据
output_data = []
for user_id, (watched_items, ratings) in sequential_data.items():
    user_id = str(user_id)
    if user_id in user_info:
        # print("yes")
        user_data = {"UserID": user_id, "history": []}
        max_len = 40
        i = 0
        for item_id, rating in zip(watched_items, ratings):
            # print(item_id, rating)
            if i<max_len:
                item_id = int(item_id)
                i+=1
                if rating > 4 and item_id in item_info:
                    item_data = {"ItemID": item_id, "title": item_info[item_id]["title"], "brand": item_info[item_id]["brand"],"category": item_info[item_id]["categories"], "price": item_info[item_id]["price"], "score": rating}
                    user_data["history"].append(item_data)
            else:
                break
        output_data.append(user_data)

# 保存结果到 JSON 文件
with open('save to output.json', 'w', encoding='utf-8') as f:
    for user_data in output_data:
        json_line = json.dumps(user_data, ensure_ascii=False)
        f.write(json_line + '\n')


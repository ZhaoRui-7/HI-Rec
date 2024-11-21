import json

# 文件路径
json_file_path = input("input file:")
mapping_file_path = './dataset/ml-1m/mapping.txt'
output_file_path = input("save:")

# 读取 JSON 文件
json_data = {}
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    for line in json_file:
        obj = json.loads(line.strip())
        json_data.update(obj)

# 读取 id1::id2 映射文件
mappings = []
with open(mapping_file_path, 'r', encoding='utf-8') as mapping_file:
    for line in mapping_file:
        id1, id2 = line.strip().split('::')
        mappings.append((id1, id2))

# 生成新的映射
new_mappings = []
for id1, id2 in mappings:
    find = 0
    for key, value in json_data.items():
        for v in range(len(value)):
            if str(id2) ==value[v][0]:
                #if [float(id2)] in value:
                #index = value.index([float(id2)])
                index = v
                new_id = key.replace('_', '')
                if len(value) > 1:
                    new_id = f"{new_id}{index}"
                new_mappings.append((id1, id2, new_id))
                find = 1
                break
        if find==1:
            break
    if find==0:
        print(id1, id2)

# 将结果写入新的文件
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for id1, id2, new_id in new_mappings:
        output_file.write(f"{id1},{id2},{new_id}\n")

print(f"新映射文件已生成: {output_file_path}")


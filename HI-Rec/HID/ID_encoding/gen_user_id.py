import json

# 读取 JSON 文件并解析每行的 JSON 对象
def read_json_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# 生成新的 ID 映射
def generate_new_id_mapping(data):
    new_id_mapping = {}
    for item in data:
        for key, value in item.items():
            # 移除 key 中的下划线
            base_key = key.replace('_', '')
            # 遍历 value 列表中的每个 ID
            for index, sublist in enumerate(value):
                old_id = sublist[0]
                if len(value) == 1:
                    new_id = base_key
                else:
                    new_id = f"{base_key}{index}"
                new_id_mapping[old_id] = new_id
    return new_id_mapping

# 将新的 ID 映射写入文件
def write_new_id_mapping(new_id_mapping, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for old_id, new_id in new_id_mapping.items():
            old_id = int(old_id)
            f.write(f"{old_id},{new_id}\n")

# 主函数
def main(input_file_path, output_file_path):
    data = read_json_file(input_file_path)
    new_id_mapping = generate_new_id_mapping(data)
    write_new_id_mapping(new_id_mapping, output_file_path)

# 示例调用
input_file_path = input("input file:")
output_file_path = input("save:")
main(input_file_path, output_file_path)


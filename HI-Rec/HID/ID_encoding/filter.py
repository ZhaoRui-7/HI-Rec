import json

# 读取原始 JSON 文件
with open('output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
s=input("save:")
# 打开一个新的文件来写入符合条件的 JSON 对象
with open(s, 'w', encoding='utf-8') as f:
    for key, value in data.items():
        if len(key) == 9:
            json.dump({key: value}, f)
            f.write('\n')  # 每个 JSON 对象占一行


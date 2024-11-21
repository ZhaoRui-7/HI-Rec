#没有把ID变成占位符
import json
import re



# 读取train_output.json文件
with open('path to output.json', 'r') as f:
    data = [json.loads(line) for line in f]

training_samples = []
# print(data)
for user_record in data:
    user_id = user_record['UserID']]
    history = user_record['history']


    for i in range(20, len(history)):
        if i>=10:
            prompt_history = history[i-10:i]
        else:
            prompt_history = history[:i]
        label = history[i]
        item_info =""
        for book in prompt_history:
            iid = item_external_ids[int(book['ItemID'])]
            s = ""
            for i in range(len(str(iid))):
                s+=f"[i_{iid[:i+1]}]"
            output_text = f"title: {book['title']}, brand: {book['brand']}, category: {book['category']}, price: {book['price']}, score: {book['score']};"
            item_info+=output_text

        s = ""
        for i in range(len(str(user_id))):
            s+=(f"[u_{user_id[:i+1]}]")
        prompt = (
            f"You are a professional book recommendation analyst. Based on the following user profile and the information about books the user has previously read, please predict what book the user might read next. \\n "
            f"The books the user has previously read are as follows:{item_info}."
        )
        lid = item_external_ids[int(label['ItemID'])]
        s = ""
        for i in range(len(str(lid))):
            s+=(f"[i_{lid[:i+1]}]")
        label = (f'The book the user might watch next is BookID={s}, book title is {label["title"]}, brand is {label["brand"]}, category is {label["category"]}, price is {label["price"]}.')
        label = ""
        training_sample = {
            "prompt": prompt,
            "label":label,
            "user_id": [int(user_record['UserID'])],
            "item_id": [item["ItemID"] for item in prompt_history]
        }

        training_samples.append(training_sample)

# 将生成的训练样本写入文件
with open('save to train_samples.json', 'w') as f:
    for sample in training_samples:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')


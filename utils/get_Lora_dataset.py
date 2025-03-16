import pandas as pd
import numpy as np

data_list = []
# 读取csv文件
for i in range(1, 7):
    try:
        # 显式指定编码为 GB18030（兼容 GBK）
        df = pd.read_csv(f'F:/desktop/{i}.csv', encoding='GB18030')[['ask', 'answer']]
        # 将ask，answer转为字典存储到data_list中，且要一对一对应
        if len(df['ask']) == len(df['answer']):
            for j in range(len(df['ask'])):
                data_list.append({'ask': df['ask'][j], 'answer': df['answer'][j]})
        else:
            print(f"文件 {i} 的行数不一致")
            continue

    except Exception as e:
        print(f"读取文件 {i} 失败: {str(e)}")
        continue

# 将文件打乱，然后转换为jsonl格式
# 其中role是user或assistant，content是对话内容，ask对应user，answer对应assistant
np.random.shuffle(data_list)
data = {"conversations": []}
for i in range(len(data_list)):
    data["conversations"].append([
        {"role": "user", "content": data_list[i]['ask']},
        {"role": "assistant", "content": data_list[i]['answer']}
    ])
# 保存文件
pd.DataFrame(data).to_json(
    "F:/desktop/myLora.jsonl",
    orient='records',
    lines=True,
    force_ascii=False,  # 关键参数：禁用 ASCII 转义
    indent=None,        # 取消缩进以压缩体积
)
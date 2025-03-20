import pandas as pd
import json
import random

def get_char_length(text):
    """计算字符长度，处理空值"""
    return 0 if pd.isna(text) else len(str(text))

# 加载并处理 distill 数据集
data2 = pd.read_json('F:/desktop/distill_r1_110k_sft.jsonl', lines=True)
data2["instruction_len"] = data2["instruction"].apply(get_char_length)
data2["output_len"] = data2["output"].apply(get_char_length)

# 过滤字符长度 <1024 的数据
filtered_data = data2[
    (data2["instruction_len"] < 1024) & 
    (data2["output_len"] < 1024)
].copy()

# 转换为目标对话格式
converted_data = []
for _, row in filtered_data.iterrows():
    # 检查 instruction 和 output 是否有效
    if pd.isna(row["instruction"]) or pd.isna(row["output"]):
        continue  # 跳过无效数据
    conversation = {
        "conversations": [
            {"role": "user", "content": str(row["instruction"])},
            {"role": "assistant", "content": str(row["output"])}
        ]
    }
    converted_data.append(conversation)

# 加载 reason 数据集
data1 = pd.read_json('F:/desktop/reason_1024.jsonl', lines=True)

# 验证 reason 数据集格式
if "conversations" not in data1.columns:
    raise ValueError("reason 数据集必须包含 'conversations' 字段")

# 合并数据集（转换为相同结构）
reason_data = data1.to_dict(orient='records')
merged_data = reason_data + converted_data

# 随机打乱数据顺序（保持每个对话完整）
random.shuffle(merged_data)

# 保存为 JSONL 文件
with open('F:/desktop/reason.jsonl', 'w', encoding='utf-8') as f:
    for entry in merged_data:
        # 格式验证：确保每个条目是 {"conversations": [...]}
        if "conversations" not in entry:
            continue
        # 检查对话轮次是否为成对 user-assistant
        valid = True
        for i, turn in enumerate(entry["conversations"]):
            if i % 2 == 0 and turn["role"] != "user":
                valid = False
            elif i % 2 == 1 and turn["role"] != "assistant":
                valid = False
        if valid:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + '\n')

print(f"处理完成！有效数据量：{len(merged_data)} 条")
import json
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass

@dataclass
class SFTDataset(Dataset):
    jsonl_path: str
    tokenizer: object
    max_length: int
    samples: list
    bos_id: list
    eos_id: list

    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_data(jsonl_path)
        # 保存bos_token和eos_token的token_id
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids

    # 返回数据长度
    def __len__(self):
        return len(self.samples)

    # 加载数据，返回数据列表
    def _load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    # 构建符合ChatML格式的对话
    def _create_chat_prompt(self, conversations):
        """构建符合ChatML格式的对话"""
        messages = []

        # 将对话转换为ChatML格式
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})

        # 使用tokenizer将对话转换为模型输入，返回的是一个字符串，包含了对话的token索引
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    # 生成动态损失掩码         
    def _generate_loss_mask(self, input_ids):
        # 初始化loss_mask，长度为input_ids的长度
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            
            # 如果遇到bos_token
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start

                # 找到对应的eos_token，标记loss_mask
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1

                # 标记loss_mask
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    # 获取数据
    def __getitem__(self, index):
        sample = self.samples[index]
        # 1.构建对话提示
        prompt = self._create_chat_prompt(sample['conversations'])

        # 2.编码文本
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]

        # 3.填充到最大长度
        if len(input_ids) < self.max_length:
            input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 4.生成动态损失掩码
        loss_mask = self._generate_loss_mask(input_ids)

        # 5. 构建模型输入：X是输入token索引，Y是标签token索引，loss_mask是损失掩码
        X = torch.tensor(input_ids[:-1], dtype=torch.long)          # X是input_ids去掉最后一个token，将根据X预测Y，所以Y是X右移一个位置
        Y = torch.tensor(input_ids[1:], dtype=torch.long)           # Y是input_ids去掉第一个token
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)   # loss_mask去掉第一个token

        # 返回模型输入，包括input_ids、labels、loss_mask
        return {
            "input_ids": X,
            "labels": Y,
            'loss_mask': loss_mask
        }
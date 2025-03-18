import json
import torch
from dataclasses import dataclass
from torch.utils.data import Dataset

@dataclass
class DPODataset(Dataset):
    jsonl_path: str
    tokenizer: object
    max_length: int
    padding: int
    bos_id: list
    eos_id: list

    def __init__(self, jsonl_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 保存padding、bos_token和eos_token的token_id
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids

        # 加载数据
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)

    # 返回数据长度
    def __len__(self):
        return len(self.data)

    # 获取数据
    def __getitem__(self, index):
        # 获取数据
        item = self.data[index]
        chosen = item['chosen']                 # 是一个 list，里面包含若干 {role, content}
        rejected = item['rejected']             # 是一个 list，里面包含若干 {role, content}
        
        # 构建符合ChatML格式的对话
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        
        # 使用tokenizer将对话转换为模型输入，返回的是一个字符串，包含了对话的token索引
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        # 生成动态损失掩码
        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)
        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)
        
        # 转换为 tensor
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        # 返回模型输入，包括对话的 token 索引、对话的 token 索引、损失掩码
        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

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
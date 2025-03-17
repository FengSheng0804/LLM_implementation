import json
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass

@dataclass
class PretrainDataset(Dataset):
    data_path: str
    tokenizer: object
    max_length: int
    samples: list

    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_data(data_path)

    # 加载数据，返回数据列表
    def _load_data(self, path):
        datas = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                datas.append(data)
        return datas
    
    # 获取数据长度
    def __len__(self):
        return len(self.samples)
    
    # 获取数据
    def __getitem__(self, index):
        """
        获取单个训练样本
        
        返回:
            {
                "input_ids": torch.Tensor,  # 输入token索引
                "attention_mask": torch.Tensor,  # 注意力掩码
                "labels": torch.Tensor  # 语言模型训练标签
            }
        """
        sample = self.samples[index]
        # 构建输入文本
        # 1. 添加特殊token：bos_token和eos_token
        text = f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"

        # 2. 编码文本：使用tokenizer将文本转换为模型可接受的输入格式
        # 返回的encoding是一个字典，包含input_ids、attention_mask等字段
        # input_ids是输入token索引，attention_mask是注意力掩码
        encoding = self.tokenizer(
            text,                           # 输入文本
            max_length=self.max_length,     # 最大长度
            padding='max_length',           # 填充到最大长度
            truncation=True,                # 截断
            return_tensors='pt'             # 返回PyTorch张量
        )

        # 3. 使用encoding获取input_ids，即输入token索引
        input_ids = encoding.input_ids.squeeze()

        # 4. 计算loss_mask：标记非填充位置，loss_mask是一个bool张量，标记非填充位置，也就是原来就有token的位置
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

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
import torch
from torch import nn


# 定义LoRA网络结构
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank                                        # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)       # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)      # 低秩矩阵B
        self.A.weight.data.normal_(mean=0.0, std=0.02)          # 矩阵A高斯初始化
        self.B.weight.data.zero_()                              # 矩阵B全0初始化

    # 前向传播
    def forward(self, x):
        # 返回矩阵A和矩阵B的乘积
        return self.B(self.A(x))

# 为模型的每个线性层应用LoRA
def apply_LoRA(model, rank=16):
    # 遍历模型的所有模块
    for name, module in model.named_modules():
        # 如果模块是线性层且权重矩阵是方阵
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            # 创建LoRA层
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            
            # 将LoRA层绑定到模块上
            setattr(module, "LoRA", lora)
            original_forward = module.forward

            # 显式绑定
            def forward_with_LoRA(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            # 重写模块的前向传播
            module.forward = forward_with_LoRA

# 加载LoRA模型
def load_LoRA(model, path):
    # 加载LoRA模型的参数
    state_dict = torch.load(path, map_location=model.device)

    # 遍历模型的所有模块
    for name, module in model.named_modules():
        # 如果模块有LoRA层
        if hasattr(module, 'LoRA'):
            # 从参数中加载LoRA层的参数
            LoRA_state = {k.replace(f'{name}.LoRA.', ''): v for k, v in state_dict.items() if f'{name}.LoRA.' in k}
            module.LoRA.load_state_dict(LoRA_state)

# 保存LoRA模型
def save_LoRA(model, path):
    state_dict = {}

    # 遍历模型的所有模块
    for name, module in model.named_modules():
        # 如果模块有LoRA层
        if hasattr(module, 'LoRA'):
            # 保存LoRA层的参数
            LoRA_state = {f'{name}.LoRA.{k}': v for k, v in module.LoRA.state_dict().items()}
            state_dict.update(LoRA_state)
    torch.save(state_dict, path)

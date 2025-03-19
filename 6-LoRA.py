import os
import time
import math
import torch
import warnings
import argparse
import torch.nn.functional as F

from model.LoRAModel import *
from model.MicroLM import MicroLM
from model.SFTDataset import SFTDataset
from model.MicroLMConfig import MicroLMConfig

from torch.optim import AdamW
from torch import distributed
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer

# 忽略警告
warnings.filterwarnings('ignore')

# 显示日志
def show_log(log):
    # 如果不是分布式训练，直接输出日志
    # 如果是分布式训练，只有rank=0的进程才会输出日志
    if not ddp or distributed.get_rank() == 0:
        print(log)

# 计算学习率
def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

# 初始化模型
def init_model(lm_config, args):
    # 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    
    # 初始化模型
    model = MicroLM(lm_config)

    # 如果使用Mixture of Experts
    moe_path = '_moe' if lm_config.use_moe else ''

    # 加载reason模型
    show_log(f'加载reason模型：{args.output_dir}/reason_{lm_config.dim}{moe_path}.pth')
    checkpoint = f'{args.output_dir}/reason_{lm_config.dim}{moe_path}.pth'
    
    # 加载reason模型的参数
    state_dict = torch.load(checkpoint, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)

    # 打印模型参数量
    show_log(f'MicroLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')

    # 将模型放到设备上，放在最后更合理
    model = model.to(args.device)
    return model, tokenizer

# 初始化分布式模式
def init_distributed_mode():
    if not ddp: 
        return
    global ddp_local_rank, DEVICE

    distributed.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])                          # 获取当前进程的rank
    ddp_local_rank = int(os.environ["LOCAL_RANK"])              # 获取当前进程的local rank
    ddp_world_size = int(os.environ["WORLD_SIZE"])              # 获取进程的数量
    DEVICE = f"cuda:{ddp_local_rank}"                           # 设置当前进程的设备
    torch.cuda.set_device(DEVICE)

# 训练一个epoch，代码和SFT几乎一致
def train_epoch(epoch, wandb, iter_per_epoch):
    loss_function = nn.CrossEntropyLoss(reduction='none')            # 使用交叉熵损失函数，reduction='none'，不对每个样本的损失求平均，保留每个样本的损失
    start_time = time.time()

    # 遍历数据加载器
    for step, batch in enumerate(train_loader):
        # 将数据放到设备上
        X, Y, loss_mask = batch["input_ids"].to(args.device), batch["labels"].to(args.device), batch["loss_mask"].to(args.device)

        # 计算学习率，根据当前epoch和step计算学习率
        learning_rate = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)

        # 设置优化器的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        # 前向传播和损失计算
        with context:
            result = model(X)                                   # 前向传播
            loss = loss_function(
                result.logits.view(-1, result.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())                                    # 计算损失
            loss = (loss * loss_mask).sum() / loss_mask.sum()   # 计算平均损失，loss_mask用于加权损失
            loss += result.aux_loss                             # 因为使用了Mixture of Experts，所以还需要加上aux_loss
            loss = loss / args.accumulation_steps               # 梯度累积

        scaler.scale(loss).backward()                           # 反向传播

        # 梯度累积与参数更新
        if (step + 1) % args.accumulation_steps == 0:           # 每accumulation_steps次迭代更新一次参数
            scaler.unscale_(optimizer)                          # 解除梯度缩放
            clip_grad_norm_(model.parameters(), args.grad_clip) # 梯度裁剪，防止梯度爆炸
            scaler.step(optimizer)                              # 更新参数
            scaler.update()                                     # 更新缩放器状态
            optimizer.zero_grad(set_to_none=True)               # 梯度清零

        # 打印日志
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            show_log(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            # 如果使用wandb，并且当前进程是主进程，记录日志
            if (wandb is not None) and (not ddp or distributed.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # 每隔save_interval保存一次模型
        if (step + 1) % args.save_interval == 0 and (not ddp or distributed.get_rank() == 0):
            model.eval()
            # 【区别1】只保存LoRA权重即可
            save_LoRA(model, f'{args.output_dir}/LoRA/{args.LoRA_name}_{lm_config.dim}.pth')
            model.train()

if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(2004)

    parser = argparse.ArgumentParser("MicroLM SFT with LoRA")
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')                             # 训练的轮数
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for data loader')                 # 数据加载器的工作线程数
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')                                        # batch size，如果过大，可能会导致内存不足
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')                              # 学习率，MiniMind设置的是5e-4
    parser.add_argument('--dim', type=int, default=512, help='Embedding dimension')                                     # 嵌入维度
    parser.add_argument('--n_layers', type=int, default=8, help='Number of layers')                                     # 层数
    parser.add_argument("--accumulation_steps", type=int, default=1)                                                    # 梯度累积步数
    parser.add_argument("--grad_clip", type=float, default=1.0)                                                         # 梯度裁剪
    parser.add_argument('--max_seq_len', type=int, default=512, help='Max sequence length')                             # 最大序列长度
    parser.add_argument('--data_path', type=str, default='./model/dataset/LoRA_identity.jsonl', help='Data path')       # 数据集的路径
    parser.add_argument("--LoRA_name", type=str, default="LoRA_identity", help="根据任务保存成LoRA_(英文/医学/心理...)")  # 保存的LoRA模型的名字
    parser.add_argument("--wandb_project", type=str, default="MicroLM-Implementation-LoRA-SFT")                         # wandb的项目名

    parser.add_argument('--use_moe', action='store_true', help='Whether to use Mixture of Experts')                     # 是否使用Mixture of Experts
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to train on')                              # 训练的设备
    parser.add_argument('--distributed', action='store_true', help='Whether to use distributed training')               # 是否使用分布式训练
    parser.add_argument("--use_wandb", action="store_true")                                                             # 是否使用wandb
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Data type')                                      # 数据类型
    parser.add_argument('--tokenizer_path', type=str, default='./model/minimind_tokenizer', help='Tokenizer to use')    # 使用的分词器
    parser.add_argument('--output_dir', type=str, default='./model_weight', help='Output directory')                    # 保存模型的路径
    parser.add_argument('--log_dir', type=str, default='./model/logs', help='Log directory')                            # 日志路径
    parser.add_argument('--log_interval', type=int, default=100, help='Log interval')                                   # 日志间隔
    parser.add_argument('--save_interval', type=int, default=1, help='Save interval')                                   # 日志保存间隔

    args = parser.parse_args()

    # 初始化模型
    lm_config = MicroLMConfig(
        dim = args.dim,
        n_layers = args.n_layers,
        max_seq_len = args.max_seq_len,
        use_moe = args.use_moe
    )

    os.makedirs(args.output_dir, exist_ok=True)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # 每次迭代的tokens数量 = 每次迭代的batch * 最大序列长度
    tokens_per_iter = args.batch_size * args.max_seq_len

    # 如果是CPU，使用nullcontext，否则使用torch.cuda.amp.autocast
    context = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast() 

    # ============================ 分布式训练 ============================
    # 是否是分布式训练
    ddp = int(os.environ.get("RANK", -1)) != -1             # 从环境变量中获取RANK，如果RANK存在，则是分布式训练，否则返回-1
    ddp_local_rank, DEVICE = 0, "cuda:0"                    # 初始化local rank和设备

    # 如果是分布式训练，初始化分布式模式
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    # ============================ wandb ============================
    # 配置wandb的运行名
    args.wandb_run_name = f"MicroLM-LoRA-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
    
    # 是否使用wandb
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # ============================ 初始化 ============================
    # 初始化模型
    model, tokenizer = init_model(lm_config, args)

    # 应用LoRA
    apply_LoRA(model)
    
    total_params = sum(p.numel() for p in model.parameters())                                       # 总参数数量
    LoRA_params_count = sum(p.numel() for name, p in model.named_parameters() if 'LoRA' in name)    # LoRA 参数数量
    
    # 打印模型参数量
    if not ddp or distributed.get_rank() == 0:
        print(f"MicroLM 总参数量: {total_params}")
        print(f"LoRA 参数量: {LoRA_params_count}")
        print(f"LoRA 参数占比: {LoRA_params_count / total_params * 100:.2f}%")

    # ============================ 模型优化 ============================
    LoRA_params = []
    for name, param in model.named_parameters():
        if 'LoRA' in name:
            LoRA_params.append(param)
        else:
            param.requires_grad = False

    # 【区别2】只对 LoRA 参数进行优化

    # 加载数据集
    train_dataset = SFTDataset(args.data_path, tokenizer, args.max_seq_len)

    # 如果是分布式训练，使用DistributedSampler
    train_sampler = DistributedSampler(train_dataset) if ddp else None

    # 初始化数据加载器
    train_loader = DataLoader(
        train_dataset,                          # 数据集
        batch_size=args.batch_size,             # batch size
        pin_memory=True,                        # 是否使用锁页内存
        drop_last=False,                        # 是否丢弃最后一个batch
        shuffle=False,                          # 是否打乱数据
        num_workers=args.num_workers,           # 数据加载器的工作
        sampler=train_sampler                   # 数据采样器
    )

    # 缩放因子，如果数据类型是float16或bfloat16，则使用GradScaler，用于进行自动混合精度（AMP）的梯度缩放
    scaler = GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))

    # 优化器
    optimizer = AdamW(LoRA_params, lr=args.learning_rate)

    iter_per_epoch = len(train_loader)                          # 每个epoch的迭代次数
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb, iter_per_epoch)               # 训练一个epoch
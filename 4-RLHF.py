import os
import time
import math
import torch
import warnings
import argparse
import torch.nn.functional as F

from model.MicroLM import MicroLM
from model.MicroLMConfig import MicroLMConfig
from model.RLHFDataset import RLHFDataset

from torch.optim import AdamW
from torch import distributed
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel
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

# 将 logits 转换为概率
# logits shape: (batch_size, seq_len, vocab_size)
# labels shape: (batch_size, seq_len)
# probs shape: (batch_size, seq_len)
def logits_to_probs(logits, labels):
    # 计算 log_softmax
    log_probs = F.log_softmax(logits, dim=2)
    # 从 log_probs 中取出 labels 对应的概率
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    
    return probs

# RLHF 损失函数
# ref_probs 和 probs 都是 shape: (batch_size, seq_len)
def RLHF_loss(ref_probs, probs, beta):
    # 计算每个样本的平均概率
    ref_probs = ref_probs.mean(dim=1)
    probs = probs.mean(dim=1)

    # 将 chosen 和 rejected 数据分开
    batch_size = ref_probs.shape[0]
    chosen_ref_probs = ref_probs[:batch_size // 2]
    reject_ref_probs = ref_probs[batch_size // 2:]
    chosen_probs = probs[:batch_size // 2]
    reject_probs = probs[batch_size // 2:]

    # 计算 pi_logratios 和 ref_logratios
    pi_logratios = chosen_probs - reject_probs
    ref_logratios = chosen_ref_probs - reject_ref_probs

    # 计算 logits 和 loss
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)

    return loss.mean()

# 初始化模型
def init_model(lm_config, args):
    # 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    
    # 初始化模型
    model = MicroLM(lm_config)

    # 如果使用Mixture of Experts
    moe_path = '_moe' if lm_config.use_moe else ''

    # 加载KD模型
    print(f'加载KD模型：{args.output_dir}/KD_{lm_config.dim}{moe_path}.pth')
    checkpoint = f'{args.output_dir}/KD_{lm_config.dim}{moe_path}.pth'

    # 加载KD模型的参数
    state_dict = torch.load(checkpoint, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)

    # 初始化参考模型
    ref_model = MicroLM(lm_config)
    ref_model.load_state_dict(state_dict, strict=False)
    ref_model.eval()
    ref_model.requires_grad_(False)

    # 打印模型参数量
    show_log(f'MicroLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')

    # 将模型和参考模型移动到设备上，放在最后更合理
    model = model.to(args.device)
    ref_model = ref_model.to(args.device)

    return model, ref_model, tokenizer

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

# 训练一个 epoch
def train_epoch(epoch, wandb, iter_per_epoch):
    start_time = time.time()

    # 遍历数据加载器
    for step, batch in enumerate(train_loader):
        # 将数据移动到设备上
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)

        # 将 chosen 和 rejected 数据合并
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # 计算当前的学习率
        learning_rate = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        
        # 更新学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        # 前向传播和损失计算
        with context:
            # 计算参考模型的输出
            with torch.no_grad():
                ref_outputs = ref_model(x)                  # 参考模型的输出
                ref_logits = ref_outputs.logits             # 参考模型的logits

            # 计算当前模型的输出
            ref_probs = logits_to_probs(ref_logits, y)
            ref_probs = ref_probs * mask                    # 参考模型的概率
            outputs = model(x)                              # 当前模型的输出
            logits = outputs.logits                         # 当前模型的logits
            probs = logits_to_probs(logits, y)
            probs = probs * mask                            # 当前模型的概率

            # 计算损失
            loss = RLHF_loss(ref_probs, probs, beta=0.1)    # RLHF损失
            loss = loss / args.accumulation_steps           # 梯度累积

        # 反向传播和梯度更新
        scaler.scale(loss).backward()

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
                wandb.log({"loss": loss,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # 每隔save_interval保存一次模型
        if (step + 1) % args.save_interval == 0 and (not ddp or distributed.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            checkpoint = f'{args.output_dir}/RLHF_{lm_config.dim}{moe_path}.pth'

            # 如果是分布式训练，则保存module的state_dict，否则保存model的state_dict
            if isinstance(model, DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            # 保存模型
            torch.save(state_dict, checkpoint)
            model.train()

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(2004)

    parser = argparse.ArgumentParser(description="MircoLM Reinforcement Learning from Human Feedback")
    parser.add_argument("--epochs", type=int, default=2)                                                # 训练的轮数
    parser.add_argument("--num_workers", type=int, default=1)                                           # 数据加载器的工作数量
    parser.add_argument("--batch_size", type=int, default=8)                                            # batch size
    # SFT阶段学习率为 「5e-6」->「5e-7」长度512，建议离线正负样本「概率」
    # 偏好对齐阶段lr <=「1e-8」长度3000，否则很容易遗忘训坏
    parser.add_argument("--learning_rate", type=float, default=1e-8)                                    # 学习率
    parser.add_argument('--max_seq_len', default=3000, type=int)                                        # 最大序列长度
    parser.add_argument('--dim', default=512, type=int)                                                 # 模型维度
    parser.add_argument("--use_wandb", action="store_true")                                             # 是否使用wandb
    parser.add_argument("--accumulation_steps", type=int, default=1)                                    # 梯度累积步数
    parser.add_argument("--grad_clip", type=float, default=1.0)                                         # 梯度裁剪

    parser.add_argument("--wandb_project", type=str, default="MicroLM-RLHF")                            # wandb项目名
    parser.add_argument("--dtype", type=str, default="bfloat16")                                        # 数据类型
    parser.add_argument("--ddp", action="store_true")                                                   # 是否分布式训练
    parser.add_argument("--device", type=str, default="cuda:0", help='Device to train on') # 设备
    parser.add_argument('--n_layers', default=8, type=int)                                              # 模型层数
    parser.add_argument('--use_moe', default=False, type=bool)                                          # 是否使用Mixture of Experts
    parser.add_argument("--data_path", type=str, default="./model/dataset/myRLHF.jsonl")                # 数据集路径
    parser.add_argument("--output_dir", type=str, default="./model_weight", help="Output directory")    # 输出目录
    parser.add_argument("--log_interval", type=int, default=100)                                        # 日志间隔
    parser.add_argument("--save_interval", type=int, default=100)                                       # 保存模型的间隔

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
    args.wandb_run_name = f"MicroLM-RLHF-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 是否使用wandb
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # ============================ 初始化 ============================
    # 初始化模型
    model, ref_model, tokenizer = init_model(lm_config, args)

    # 加载数据集
    train_dataset = RLHFDataset(args.data_path, tokenizer, args.max_seq_len)

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

    # 初始化优化器
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # 如果是分布式训练，使用DistributedDataParallel
    if ddp:
        # 忽略pos_cis参数
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)                          # 每个epoch的迭代次数
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb, iter_per_epoch)               # 训练一个epoch
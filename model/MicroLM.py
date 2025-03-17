import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MicroLMConfig import MicroLMConfig
from transformers import PreTrainedModel
from typing import List, Optional, Tuple
from transformers.modeling_outputs import CausalLMOutputWithPast

# 预计算位置编码
def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return pos_cis

# RoPE，旋转位置编码
def apply_rotary_emb(q, k, pos_cis):
    # 将位置编码与输入张量拼接
    def unite_shape(pos_cis, x):
        # 获取输入张量的维度
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])

        # 创建一个新的形状列表，除了第1维和最后一维，其他维度都设置为1
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]

        # 将位置编码调整为新的形状
        return pos_cis.view(*shape)

    # 将q、k转换为复数形式
    q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    
    # 调整位置编码的形状以匹配查询张量
    pos_cis = unite_shape(pos_cis, q_)

    # 应用旋转位置编码到q、k
    q_out = torch.view_as_real(q_ * pos_cis).flatten(3)
    k_out = torch.view_as_real(k_ * pos_cis).flatten(3)
    
    return q_out.type_as(q), k_out.type_as(k)

# 拓展key和value的维度
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )

# RMSNorm，用于替代LayerNorm，是一种新的归一化方法，它是在LayerNorm的基础上进行改进的，
# RMSNorm的计算公式为：RMSNorm(x) = x * (x.pow(2).mean(dim) + eps).rsqrt()
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * (x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)

# 注意力层
class Attention(nn.Module):
    def __init__(self, args: MicroLMConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads      # 多头注意力头数，如果args.n_kv_heads为None，则使用args.n_heads
        assert args.n_heads % self.n_kv_heads == 0                                          # 多头注意力头数必须能被key和value的头数整除
        self.n_local_heads = args.n_heads                                                   # 总query的头数
        self.n_local_kv_heads = self.n_kv_heads                                             # key和value的头数
        
        self.head_dim = args.dim // args.n_heads                                            # 每个头的维度
        self.n_rep = self.n_local_heads // self.n_local_kv_heads                            # 每个key和value头的重复次数

        # query、key、value的线性变换，投影到多头注意力的维度
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)             # 查询投影
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)          # 键投影
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)          # 值投影
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)             # 输出投影

        # 正则化层
        self.attn_dropout = nn.Dropout(args.dropout)                                           # 注意力权重dropout
        self.resid_dropout = nn.Dropout(args.dropout)                                          # 残差连接前dropout

        # 参数配置
        self.dropout = args.dropout
        # 如果torch.nn.functional中有scaled_dot_product_attention函数，且args.flash_attn为True，则使用flash attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

        # 自回归掩码
        # 为了保证模型的自回归性，我们需要在每个位置只能看到它之前的位置，不看到后面的位置，因此我们需要一个上三角的掩码
        # 这里我们使用一个非常大的负数来填充掩码，这样在softmax之后，这些位置的概率就会接近于0
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)                                                 # 上三角掩码
        
        # 注册掩码，存放在模型的buffer中，避免在模型的forward中重复创建
        self.register_buffer("mask", mask, persistent=False)

    def forward(self,
                x: torch.Tensor,
                pos_cis: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False
                ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: 输入张量，形状为[batch_size, seq_len, dim]
            pos_cis: 位置编码，形状为[seq_len, dim]
            past_kv: 过去的key和value，形状为[2, batch_size, n_kv_heads, seq_len, head_dim]
            use_cache: 是否使用缓存
        Returns:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]
        """
        # 获取batch_size和seq_len
        batch_size, seq_len, _ = x.shape

        # 线性投影得到Q/K/V
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # 将多头拆分：[batch_size, seq_len, n_heads, head_dim]
        q = q.view(batch_size, seq_len, self.n_local_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)

        # 应用旋转位置编码RoPE
        q, k = apply_rotary_emb(q, k, pos_cis)

        # 如果使用缓存，则将过去的key和value拼接到当前的key和value上
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)                                    # 拼接历史Key
            v = torch.cat([past_key_value[1], v], dim=1)                                    # 拼接历史Value
        past_key_value = (k, v) if use_cache else None

        # 调整维度为 (batch_size, n_heads, seq_len, head_dim)，使得多头可以并行计算
        q = q.transpose(1, 2)
        k = repeat_kv(k, self.n_rep).transpose(1, 2)                                        # 拓展key的维度
        v = repeat_kv(v, self.n_rep).transpose(1, 2)                                        # 拓展value的维度

        # Flash Attention加速模式
        if self.flash and seq_len != 1:
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True                                                              # 自动应用因果掩码
            )
        else:
            # 手动计算注意力
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores += self.mask[:, :, :seq_len, :seq_len]                                   # 应用因果掩码
            scores = F.softmax(scores.float(), dim=-1).type_as(q)
            scores = self.attn_dropout(scores)
            output = scores @ v

        # 输出处理
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.resid_dropout(self.wo(output))                                        # 输出投影+Dropout
        return output, past_key_value

# FeedForward层
class FeedForward(nn.Module):
    def __init__(self, config: MicroLMConfig):
        super().__init__()
        # 动态计算隐藏层维度
        if config.hidden_dim is None:
            hidden_dim = 4 * config.dim                                                     # 初始设定：4倍模型维度
            hidden_dim = int(2 * hidden_dim / 3)                                            # 缩放因子：2/3

            # 对齐到最近的multiple_of倍数
            config.hidden_dim = config.multiple_of * ( 
                (hidden_dim + config.multiple_of - 1) // config.multiple_of
            )
        
        # 定义线性层
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)                      # 将输入升维到隐藏空间（类似标准FFN的第一层）
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)                      # 将门控结果降维回原始维度
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)                      # 生成门控信号（与w1并行但独立）
        self.dropout = nn.Dropout(config.dropout)                                           # 输出正则化

    def forward(self, x):
        # 公式：FFN(x) = Dropout(W2(SiLU(W1(x)) ⊙ W3(x)))
        gate = F.silu(self.w1(x))                                                           # SiLU激活函数
        modulated = gate * self.w3(x)                                                       # 逐元素相乘（门控）
        output = self.w2(modulated)                                                         # 降维投影
        return self.dropout(output)                                                         # 正则化输出

class MoEGate(nn.Module):
    def __init__(self, config: MicroLMConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok                                             # 每个token选择的专家数
        self.n_routed_experts = config.n_routed_experts                                     # 专家总数
        
        self.scoring_func = config.scoring_func                                             # 分数计算函数（当前仅支持softmax）
        self.alpha = config.aux_loss_alpha                                                  # 辅助损失系数
        self.seq_aux = config.seq_aux                                                       # 是否使用序列级辅助损失
        self.norm_topk_prob = config.norm_topk_prob                                         # 是否对top_k权重归一化
        self.gating_dim = config.dim                                                        # 门控维度
        
        # 可学习参数：专家选择权重矩阵
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    # 重置参数
    def reset_parameters(self) -> None:
        import torch.nn.init as init
        # 使用Kaiming初始化，适应后续的SiLU等激活函数
        init.kaiming_uniform_(self.weight, a=math.sqrt(5)) 

    def forward(self, hidden_states):
            # 输入形状: (batch_size, seq_len, hidden_dim)
            batch_size, seq_len, h = hidden_states.shape
            
            # 将输入展平为 (batch_size*seq_len, hidden_dim)
            hidden_states = hidden_states.view(-1, h)
            
            # 计算原始分数 logits: [batch_size*seq_len, n_experts]
            logits = F.linear(hidden_states, self.weight, None)                                 # 无偏置项
            
            # 将分数转换为概率分布
            if self.scoring_func == 'softmax':
                scores = logits.softmax(dim=-1)                                                 # 按最后一个维度做Softmax
            else:
                raise NotImplementedError(f"不支持的专家评分函数: {self.scoring_func}")

            # 选择Top-K专家 [value, indices]
            topk_weight, topk_idx = torch.topk(
                scores, 
                k=self.top_k, 
                dim=-1, 
                sorted=False                                                                    # 不排序以提高效率
            )

            # 权重归一化（当选择多个专家时）
            if self.top_k > 1 and self.norm_topk_prob:
                denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20                     # 防止除零
                topk_weight = topk_weight / denominator                                         # 归一化权重和为1

            # 辅助损失计算（仅在训练时且alpha>0时激活）
            if self.training and self.alpha > 0.0:
                scores_for_aux = scores                                                         # 原始概率分数
                
                # 序列级辅助损失（平衡每个序列内的专家使用）
                if self.seq_aux:
                    scores_for_seq_aux = scores_for_aux.view(batch_size, seq_len, -1)                  # [batch_size, seq_len, n_experts]
                    ce = torch.zeros(batch_size, self.n_routed_experts, device=hidden_states.device)
                    
                    # 统计每个专家在batch中被选中的次数
                    ce.scatter_add_(
                        1, 
                        topk_idx.view(batch_size, -1),                                                 # [batch_size, seq_len*top_k]
                        torch.ones(batch_size, seq_len * self.top_k, device=hidden_states.device)
                    ).div_(seq_len * self.top_k / self.n_routed_experts)                        # 归一化为频率
                    
                    # 计算损失：专家使用频率与平均分数的乘积
                    aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
                
                # Token级辅助损失（全局平衡专家使用）
                else:
                    # 生成one-hot编码的专家选择掩码
                    mask_ce = F.one_hot(
                        topk_idx.view(-1),                                                      # 展平所有选择的专家索引
                        num_classes=self.n_routed_experts
                    )
                    ce = mask_ce.float().mean(0)                                                # 计算每个专家的平均被选概率
                    Pi = scores_for_aux.mean(0)                                                 # 计算每个专家的平均激活概率
                    fi = ce * self.n_routed_experts                                             # 理想均匀分布下的期望值
                    
                    # 计算损失：实际分布与理想分布的协方差
                    aux_loss = (Pi * fi).sum() * self.alpha
            else:
                aux_loss = 0                                                                    # 无辅助损失

            return topk_idx, topk_weight, aux_loss                                              # 返回专家索引、权重和损失

class MOEFeedForward(nn.Module):
    def __init__(self, config: MicroLMConfig):
        super().__init__()
        self.config = config
        # 初始化专家池：创建n_routed_experts个独立的前馈网络
        self.experts = nn.ModuleList([
            FeedForward(config) for _ in range(config.n_routed_experts)
        ])
        # 初始化门控模块（负责路由选择）
        self.gate = MoEGate(config)
        # 可选共享专家（用于增强基础能力）
        if config.n_shared_experts is not None:
            self.shared_experts = FeedForward(config)

    def forward(self, x):
        identity = x                                                                        # 保留原始输入用于残差连接
        orig_shape = x.shape                                                                # 记录原始形状 [batch, seq_len, dim]
        batch_size, seq_len, _ = x.shape
        
        # 门控计算：获取top_k专家索引、权重和辅助损失
        topk_idx, topk_weight, aux_loss = self.gate(x)
        
        # 展平输入：将输入转换为 [batch*seq_len, dim]
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)                                                   # 展平为 [batch*seq_len*top_k]
        
        if self.training:
            # 训练模式：复制输入数据以匹配专家处理需求
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)                                    # 预分配结果张量
            
            # 分布式处理：每个专家独立处理分配给它的token
            for i, expert in enumerate(self.experts):
                mask = (flat_topk_idx == i)                                                 # 当前专家处理的token掩码
                if mask.any():
                    y[mask] = expert(x[mask]).to(y.dtype)                                   # 类型转换确保计算兼容性
            
            # 加权聚合：将多专家结果按权重合并 [batch*seq_len*top_k, dim] -> [batch*seq_len, dim]
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)                                                         # 恢复原始形状 [batch, seq_len, dim]
        else:
            # 推理模式：优化计算路径，减少冗余操作
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        
        # 叠加共享专家结果（如果配置）
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        
        self.aux_loss = aux_loss                                                            # 保存辅助损失用于反向传播
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        # 初始化结果缓存张量
        expert_cache = torch.zeros_like(x)
        # 按专家索引排序，优化计算顺序
        idxs = flat_expert_indices.argsort()
        # 统计每个专家处理的token数量（累计和形式）
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 计算原始token索引（处理重复选择的token）
        token_idxs = idxs // self.config.num_experts_per_tok
        
        # 分块处理：每个专家依次处理分配到的token
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:                                                        # 跳过无token的专家
                continue
                
            expert = self.experts[i]                                                        # 当前专家
            exp_token_idx = token_idxs[start_idx:end_idx]                                   # 当前专家处理的token索引
            expert_tokens = x[exp_token_idx]                                                # 提取对应token
            
            # 专家计算并加权
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            
            # 累加到结果缓存（使用scatter_add避免重复计算）
            expert_cache.scatter_add_(
                0, 
                exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),                           # 目标索引
                expert_out                                                                  # 待添加数据
            )
            
        return expert_cache                                                                 # 返回聚合结果

class MircoLLMBlock(nn.Module):
    def __init__(self, layer_id: int, config: MicroLMConfig):
        super().__init__()
        # 初始化多头注意力参数
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        
        # 核心模块定义
        self.attention = Attention(config)                                                  # 自注意力模块
        self.layer_id = layer_id                                                            # 当前层编号
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)                      # 注意力前归一化
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)                            # FFN前归一化

        # 动态选择前馈类型（普通FFN或MoE）
        self.feed_forward = (
            FeedForward(config) 
            if not config.use_moe 
            else MOEFeedForward(config)
        )

    def forward(self, x, pos_cis, past_key_value=None, use_cache=False):
        # 自注意力计算（带归一化）
        h_attn, past_kv = self.attention(
            self.attention_norm(x),                                                         # 前置归一化
            pos_cis,                                                                        # 预计算的位置编码
            past_key_value=past_key_value,                                                  # 历史KV缓存
            use_cache=use_cache                                                             # 是否缓存当前KV
        )
        # 残差连接
        h = x + h_attn                                                                      # 注意力残差
        
        # 前馈网络计算（带归一化）
        out = h + self.feed_forward(self.ffn_norm(h))                                       # FFN残差
        return out, past_kv                                                                 # 返回输出和当前层KV缓存
    
class MicroLM(PreTrainedModel):
    # 指定配置类为自定义的LMConfig
    config_class = MicroLMConfig

    def __init__(self, params: MicroLMConfig = None):
        # 初始化模型参数
        self.params = params or MicroLMConfig()                                             # 若未提供配置则用默认配置
        super().__init__(self.params)                                                       # 调用父类PreTrainedModel的初始化
        # 基础参数设置
        self.vocab_size = params.vocab_size                                                 # 词表大小
        self.n_layers = params.n_layers                                                     # Transformer层数
        
        # 词嵌入层定义
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        # 正则化层
        self.dropout = nn.Dropout(params.dropout)
        
        # 构建Transformer层堆栈
        self.layers = nn.ModuleList([
            MircoLLMBlock(l, params) for l in range(self.n_layers)
        ])
        
        # 最终层归一化
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        # 输出投影层（无偏置）
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        # 权重绑定：词嵌入与输出层共享权重矩阵
        self.tok_embeddings.weight = self.output.weight
        
        # 注册预计算的旋转位置编码（RoPE）
        self.register_buffer(
            "pos_cis",
            precompute_pos_cis(
                dim=params.dim // params.n_heads,                                           # 单头维度
                theta=params.rope_theta                                                     # RoPE基频参数
            ),
            persistent=False                                                                # 不保存到模型文件
        )
        # 初始化输出容器
        self.OUT = CausalLMOutputWithPast()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **args
    ):
        # 初始化KV缓存（每层初始化为None）
        past_key_values = past_key_values or [None] * len(self.layers)
        # 获取起始位置（用于流式生成）
        start_pos = args.get('start_pos', 0)
        
        # 词嵌入与正则化
        h = self.dropout(self.tok_embeddings(input_ids))
        # 截取当前位置编码（支持长文本生成）
        pos_cis = self.pos_cis[start_pos : start_pos + input_ids.size(1)]
        
        # 逐层处理
        past_kvs = []
        for l, layer in enumerate(self.layers):
            h, past_kv = layer(
                h, 
                pos_cis,
                past_key_value=past_key_values[l],                                          # 传入该层历史KV
                use_cache=use_cache                                                         # 是否返回当前层KV
            )
            past_kvs.append(past_kv)                                                        # 收集各层新KV缓存
        
        # 最终归一化与输出投影
        logits = self.output(self.norm(h))
        # 计算所有MoE层的辅助损失总和
        aux_loss = sum(
            l.feed_forward.aux_loss 
            for l in self.layers 
            if isinstance(l.feed_forward, MOEFeedForward)                                   # 仅统计MoE层
        )
        
        # 组装输出结果到指定容器
        self.OUT.__setitem__('logits', logits)                                              # 预测logits
        self.OUT.__setitem__('aux_loss', aux_loss)                                          # MoE路由辅助损失
        self.OUT.__setitem__('past_key_values', past_kvs)                                   # 各层新KV缓存
        return self.OUT

    @torch.inference_mode()                                                                 # 禁用梯度计算以提升推理效率
    def generate(
        self, 
        input_ids, 
        eos_token_id=2,                                                                     # 终止符ID（示例值）
        max_new_tokens=1024,                                                                # 最大生成长度
        temperature=0.75,                                                                   # 温度参数（>1更随机，<1更确定）
        top_p=0.90,                                                                         # 核采样累积概率阈值
        stream=False,                                                                       # 是否流式输出
        rp=1.,                                                                              # 重复惩罚系数（>1降低重复概率）
        use_cache=True,                                                                     # 是否使用KV缓存加速
        pad_token_id=0,                                                                     # 填充符ID
        **args
    ):
        # 流式生成模式（逐token返回）
        if stream:
            return self._stream(input_ids, eos_token_id, max_new_tokens, 
                               temperature, top_p, rp, use_cache, **args)

        # 批处理生成模式
        generated = []
        # 遍历批次中的每个样本（支持批量生成）
        for i in range(input_ids.size(0)):
            # 去除当前样本的填充token
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)
            # 调用流式生成函数
            out = self._stream(non_pad, eos_token_id, max_new_tokens, 
                              temperature, top_p, rp, use_cache, **args)
            # 收集生成结果
            tokens_list = [tokens[:, -1:] for tokens in out]
            gen = torch.cat(tokens_list, dim=-1) if tokens_list else non_pad
            # 拼接原始输入与生成结果
            full_sequence = torch.cat([non_pad, gen], dim=-1)
            generated.append(full_sequence)
        
        # 对齐批次中各样本长度（填充处理）
        max_length = max(seq.size(1) for seq in generated)
        generated = [
            torch.cat([
                seq, 
                # 填充右侧至最大长度
                torch.full(
                    (1, max_length - seq.size(1)), 
                    pad_token_id, 
                    dtype=seq.dtype, 
                    device=seq.device
                )
            ], dim=-1) 
            for seq in generated
        ]
        return torch.cat(generated, dim=0)                                                  # 合并为批次张量

    def _stream(
        self, 
        input_ids, 
        eos_token_id, 
        max_new_tokens, 
        temperature, 
        top_p, 
        rp, 
        use_cache, 
        **args
    ):
        # 初始化生成状态
        start = input_ids.shape[1]                                                          # 记录初始输入长度
        first_seq = True                                                                    # 是否为首次推理
        past_kvs = None                                                                     # KV缓存初始化
        
        # 自回归生成循环
        while input_ids.shape[1] < max_new_tokens - 1:
            # 首次推理或禁用缓存时处理全部输入
            if first_seq or not use_cache:
                out = self(
                    input_ids, 
                    past_key_values=past_kvs, 
                    use_cache=use_cache, 
                    **args
                )
                first_seq = False
            else:
                # 非首次推理且启用缓存时，仅处理最后一个token
                out = self(
                    input_ids[:, -1:],                                                      # 仅取最后一个token
                    past_key_values=past_kvs, 
                    use_cache=use_cache,
                    start_pos=input_ids.shape[1] - 1,                                       # 指定当前位置
                    **args
                )
            
            # 提取当前步骤的logits和更新后的KV缓存
            logits = out.logits[:, -1, :]                                                   # 取最后一个位置的logits
            past_kvs = out.past_key_values
            
            # 重复惩罚：降低已出现token的概率
            appeared_tokens = list(set(input_ids.tolist()[0]))                              # 已生成token去重
            logits[:, appeared_tokens] /= rp                                                # 调整对应位置的logits
            
            # 温度调节：控制概率分布尖锐程度
            logits /= (temperature + 1e-9)                                                  # 防止除零错误
            
            # Top-p（核）采样
            if top_p < 1.0:
                # 按概率降序排列
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # 创建保留掩码（累积概率<=top_p的最小集合）
                sorted_indices_to_remove = cumulative_probs > top_p
                # 右移一位，保留第一个超过阈值的token
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False                                      # 至少保留一个token
                
                # 将掩码映射回原始索引顺序
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, 
                    sorted_indices, 
                    sorted_indices_to_remove
                )
                # 屏蔽低概率token
                logits[indices_to_remove] = -float('Inf')
            
            # 从调整后的分布中采样
            probs = F.softmax(logits, dim=-1)
            input_ids_next = torch.multinomial(probs, num_samples=1)
            
            # 拼接新生成的token
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)
            
            # 流式输出当前结果（排除初始输入）
            yield input_ids[:, start:]
            
            # 终止条件检查
            if input_ids_next.item() == eos_token_id:
                break
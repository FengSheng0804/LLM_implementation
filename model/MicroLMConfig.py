from transformers import PretrainedConfig

class MicroLMConfig(PretrainedConfig):
    model_type = "MicroLM"

    def __init__(
            self,
            dim: int = 512,                                         # 模型维度
            n_layers: int = 8,                                      # 层数
            n_heads: int = 8,                                       # 多头注意力头数
            n_kv_heads: int = 2,                                    # 多头注意力中的key和value头数
            vocab_size: int = 6400,                                 # 词表大小
            hidden_dim: int = None,                                 # 隐藏层维度
            multiple_of: int = 64,                                  # 输入维度的倍数
            norm_eps: float = 1e-4,                                 # 归一化层的epsilon
            max_seq_len: int = 8192,                                # 最大序列长度
            rope_theta: int = 1e6,                                  # ROPE的theta参数
            dropout: float = 0.0,                                   # dropout概率
            flash_attn: bool = True,                                # 是否使用flash attention
            #####################################
            # 这里是Mixture of Experts的具体配置 #
            # 当use_moe为false时，以下无效       #
            #####################################
            use_moe: bool = False,                                  # 是否使用Mixture of Experts
            num_experts_per_tok: int = 2,                           # 每个token选择的专家数量
            n_routed_experts: int = 4,                              # 总的专家数量
            n_shared_experts: bool = True,                          # 共享专家
            scoring_func: str = 'softmax',                          # 评分函数，默认为'softmax'
            aux_loss_alpha: float = 0.1,                            # 辅助损失的alpha参数
            seq_aux: bool = True,                                   # 是否在序列级别上计算辅助损失
            norm_topk_prob: bool = True,                            # 是否标准化top-k概率
            **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.flash_attn = flash_attn
        #####################################
        # 这里是Mixture of Experts的具体配置 #
        # 当use_moe为false时，以下无效       #
        #####################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        super().__init__(**kwargs)

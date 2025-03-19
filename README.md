# LLM_implementation

从0到1实现轻量级大语言模型，本项目参考自github中的[MiniMind](https://github.com/jingyaogong/minimind)项目

## 第一步：从网上获取数据集

### 预训练数据集

对于预训练数据集，匠数大模型SFT数据集是一个完整、格式统一、安全的大模型训练和研究资源。从网络上的公开数据源收集并整理了大量开源数据集，对其进行了格式统一，数据清洗， 包含10M条数据的中文数据集和包含2M条数据的英文数据集。但是官方整理得到的数据集的格式比较混乱，因此我们直接使用MiniMind官方提供的[pretrain_hq数据集](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/file/view/master?id=68909&status=2&fileName=pretrain_hq.jsonl)训练即可。

数据集的格式如下：

```json
{"text": "<s>鉴别一组中文文章的风格和特点，例如官方、口语、文言等。需要提供样例文章才能准确鉴别不同的风格和特点。</s> <s>好的，现在帮我查一下今天的天气怎么样?今天的天气依据地区而异。请问你需要我帮你查询哪个地区的天气呢？</s> <s>打开闹钟功能，定一个明天早上七点的闹钟。好的，我已经帮您打开闹钟功能，闹钟将在明天早上七点准时响起。</s> <s>为以下场景写一句话描述：一个孤独的老人坐在公园长椅上看着远处。一位孤独的老人坐在公园长椅上凝视远方。</s> <s>非常感谢你的回答。请告诉我，这些数据是关于什么主题的？这些数据是关于不同年龄段的男女人口比例分布的。</s> <s>帮我想一个有趣的标题。这个挺有趣的：\"如何成为一名成功的魔术师\" 调皮的标题往往会吸引读者的注意力。</s> <s>回答一个问题，地球的半径是多少？地球的平均半径约为6371公里，这是地球自赤道到两极的距离的平均值。</s> <s>识别文本中的语气，并将其分类为喜悦、悲伤、惊异等。\n文本：“今天是我的生日！”这个文本的语气是喜悦。</s>"}
```

### 有监督微调数据集

对于有监督微调数据集，同样的我们使用MiniMind官方整理的[sft_mini_512数据集](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/file/view/master?id=68909&status=2&fileName=sft_mini_512.jsonl)，该数据集整合自匠数科技SFT数据+Qwen2.5蒸馏数据，每条数据字符最大长度为512，因此训练时应将max_seq_len设置为512。

```json
{"conversations": 
 	[
        {
            "role": "user", 
            "content": "请告诉我在中国古代的“四大发明”是什么？"
        }, 
        {
            "role": "assistant", 
            "content": "中国古代的“四大发明”是指造纸术、印刷术、火药和指南针..."
        }
    ]
}
```

### 知识蒸馏数据集

将教师模型（Teacher，如GPT-4）的知识（如输出概率分布、隐藏层特征）迁移到学生模型（Student，如TinyBERT），显著降低模型参数量和计算资源需求，同时保持性能。学生模型通过模仿教师模型的输出逻辑（如文本生成风格、逻辑推理能力），减少推理时的计算量（如从GPU到CPU部署）。

对于知识蒸馏数据集，我们采用MiniMind官方整理的[sft_1024数据集](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/file/view/master?id=68909&status=2&fileName=sft_1024.jsonl)，MiniMind官方将这部分数据进一步清洗，把总长度`<1024`的部分导出为`sft_1024.jsonl`(~5.5GB)，用大模型对话数据直接进行sft就属于“黑盒蒸馏”的范畴。

数据集的格式如下：

```json
{
    "conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "再见"},
        {"role": "assistant", "content": "再见！"}
    ]
}
```

### 人类反馈强化学习数据集

对于人类反馈强化学习数据集，我们采用整合自[MagpieLM]([MagpieLM-DPO-Data-v0.1 · 数据集](https://www.modelscope.cn/datasets/Magpie-Align/MagpieLM-DPO-Data-v0.1/files))官方的数据集，由于该数据集是parquet格式的，为了符合MiniMind的数据集，我们需要将数据集进行处理，得到jsonl格式的数据集。数据集的格式如下：

| uuid (Value)                         | instruction (Value)                                                                                                                                      | chosen (Value)                                                                                                                              | rejected (Value)                                                                                                                            |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| bf73e2af-0c8e-5356-bb1e-aac105bd3da0 | ### Edge Cases for Optional Chaining #### Problem You have an object that may or may not contain a certain path. You want to safely navigate this pat... | [{"content":"### Edge Cases for Optional Chaining\n#### Problem\nYou have an object that may or may not contain a certain path. You want... | [{"content":"### Edge Cases for Optional Chaining\n#### Problem\nYou have an object that may or may not contain a certain path. You want... |

我们只需要使用里面的chosen和rejected数据即可，因此经过整理后，得到的数据的格式如下（同MiniMind）：

```json
{
  "chosen": [
    {"content": "Q", "role": "user"}, 
    {"content": "good answer", "role": "assistant"}
  ], 
  "rejected": [
    {"content": "Q", "role": "user"}, 
    {"content": "bad answer", "role": "assistant"}
  ]
}
```

### 参数高效微调数据集

LoRA 是一种**参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）**技术，专为大型语言模型（LLM）设计。其核心是通过**低秩矩阵分解**，仅微调模型的一小部分参数（通常 <1%），即可达到接近全参数微调的效果。

参数高效微调的数据集我们自己收集，我们使用的是[Chinese medical dialogue data 中文医疗问答数据集]([Toyhom/Chinese-medical-dialogue-data: Chinese medical dialogue data 中文医疗对话数据集](https://github.com/Toyhom/Chinese-medical-dialogue-data/tree/master))，这个数据集共包含以下内容：**<Andriatria_男科>** 94596个问答对 **<IM_内科>** 220606个问答对 **<OAGD_妇产科>** 183751个问答对 **<Oncology_肿瘤科>** 75553个问答对 **<Pediatric_儿科>** 101602个问答对 **<Surgical_外科>** 115991个问答对 总计 792099条数据，其中的内容格式如下：

| department |                  title | question                                                                   | answer                                                                                                                                                                                                                                                                                               |
| ---------- | ---------------------: | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 心血管科   | 高血压患者能吃党参吗？ | 我有高血压这两天女婿来的时候给我拿了些党参泡水喝，您好高血压可以吃党参吗？ | 高血压病人可以口服党参的。党参有降血脂，降血压的作用，可以彻底消除血液中的垃圾，从而对冠心病以及心血管疾病的患者都有一定的稳定预防工作作用，因此平时口服党参能远离三高的危害。另外党参除了益气养血，降低中枢神经作用，调整消化系统功能，健脾补肺的功能。感谢您的进行咨询，期望我的解释对你有所帮助。 |
| 消化科     |     哪家医院能治胃反流 | 烧心，打隔，咳嗽低烧，以有4年多                                            | 建议你用奥美拉唑同时，加用吗丁啉或莫沙必利或援生力维，另外还可以加用达喜片                                                                                                                                                                                                                           |

我们只是用其中的question和answer部分即可，然后整理成如下的jsonl形式：

```json
{
    "conversations": [
        {"role": "user","content": "请问颈椎病的人枕头多高才最好？"}, 
        {"role": "assistant","content": "颈椎病患者选择枕头的高度应该根据..."}
    ]
}
{	
    "conversations": [
        {"role": "user","content": "请问xxx"},
        {"role": "assistant","content": "xxx..."}
    ]
}
```

### 推理模型数据集

推理模型的数据集我们依旧采用MiniMind官方提供的[r1_mix_1024数据集](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/file/view/master?id=68909&status=2&fileName=r1_mix_1024.jsonl)，其数据的格式如下：其中assitant的回答包括**<think>**和**<answer>**两部分

```json
{
    "conversations": [
        {"role": "user", "content": "请用一段话描述阿里巴巴集团的企业文化。"}, 
        {"role": "assistant", "content": "<think>\n嗯，用户让我用一段话描述阿里巴巴集团的企业文化...\n</think>\n<answer>\n阿里巴巴集团的企业文化以战略协作为核心，...\n</answer>"}
    ]
}
```

## 第二步：获取Tokenizer

我们使用MiniMind官方提供的Tokenizer，其可以通过data_process.py得到，因此不再重复训练。

项目自定义的Tokenizer模型文件。

- model/minimind_tokenizer/merges.txt
  merges文件存放的是训练tokenizer阶段所得到的合并词表结果，就是tokenizer.json中，model.merges下的内容。
- model/minimind_tokenizer/tokenizer_config.json
  分词器的配置信息，定义了分词器的版本、额外添加的标记（tokens）、结构/代码和模型参数等信息，比如tokenizer_class指定使用的分词器类名以及model_max_length指定模型能够处理的最大序列长度 和 bos_token指定句首的标记等内容。
- model/minimind_tokenizer/tokenizer.json
  最终的分词器模型文件，包含了分词器的版本号、分词器的截断、填充策略、特殊标记、文本归一化的函数、预分词的策略或方法、分词器模型的类型、词汇表（vocab）和合并规则（merges）等信息。
- model/minimind_tokenizer/vocab.json
  词表文件，就是tokenizer.json中，model.vocab下的内容。

## 第三步：开始训练

### 预训练

我在AutoDL平台租用了一台拥有RTX 4090 * 1服务器，在执行的时候我们选择的参数是`epoch=2`，`batch_size=32`，`max_seq_len=512`，`learning_rate=5e-4`，`dim=512`，在运行了大概2个小时后，训练完成，训练时的损失曲线如下图所示：

![pretrain_loss_curve](.\images\pretrain_loss_curve.png)

最终测试效果如下：

可以看到模型在训练中已经具备了基本的知识储备，但是还有点语无伦次的情况出现。

![image-20250318105115208](.\images\image-20250318105115208.png)

### 有监督微调-sft_mini_512

在执行完预训练后，我们使用sft_mini_512.jsonl数据集进行训练，在执行的时候我们选择的参数是`epoch=2`，`batch_size=32`，`max_seq_len=512`，`learning_rate=5e-5`，`dim=512`，在运行了大概2个小时后，训练完成，训练时的损失曲线如下图所示：

![sft_loss_curve](.\images\sft_loss_curve.png)

最终测试效果如下：

可以看到，模型已经能很好的回答我们的问题，但是有事会出现一些错误情况，比如图片中的最后一个问题。同时在问题4中，由于模型还没有经过RLHF训练，导致回答的相对笼统。

![image-20250318105640404](.\images\image-20250318105640404.png)

### 知识蒸馏

在执行完有监督训练-sft_mini_512后，我们使用sft_mini_512.jsonl数据集继续进行知识蒸馏，在执行的时候我们选择的参数是`epoch=2`，`batch_size=64`，`max_seq_len=1024`，`learning_rate=5e-5`，`dim=512`，由于数据集比较大，大约是5G左右，所以我们选择了单机双卡服务器，通过命令`torchrun --nproc_per_node 2 3-KD.py --use_wandb`在运行了大概6个小时后，训练完成，训练时的损失曲线如下图所示：

![KD_loss_curve](.\images\KD_loss_curve.png)

最终的测试结果如下：

可以看到经过知识蒸馏后的模型回答问题更加精确，并且可以分条回答，具有较好的结果。

![image-20250319083554257](.\images\image-20250319083554257.png)







### 人类反馈强化学习

在执行完知识蒸馏后，我们使用自行收集得到的myRLHF.jsonl数据集进行人类反馈强化学习（RLHF），此处我们使用的是DPO（Direct Preference Optimization），与PPO(Proximal Policy Optimization)这种需要奖励模型、价值模型的RL算法不同； DPO通过推导PPO奖励模型的显式解，把在线奖励模型换成离线数据，Ref模型输出可以提前保存。

在执行时我们选择的参数是`epoch=2`，`batch_size=8`，`max_seq_len=3000`，`learning_rate=1e-8`，`dim=512`，在训练时，我们选择了单机4卡服务器，通过命令`torchrun --nproc_per_node 4 4-RLHF --use_wandb`在运行了大概1个小时后，训练完成，训练时的损失曲线如下图所示：

![RLHF_loss_curve](.\images\RLHF_loss_curve.png)

最终的测试结果如下：

![image-20250319083355281](.\images\image-20250319083355281.png)

### 参数高效微调

在执行完人类反馈强化学习后，我们使用自行收集得到的LoRA_medical数据集进行参数高效微调（LoRA），我们使用到的参数是



### 推理模型训练

在执行完参数高效微调后，我们使用MiniMind官方提供的r1_mix_1024.jsonl数据集进行训练，




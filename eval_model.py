# 测评模型性能，lora模型的权重文件在out/lora目录下

import argparse
import random
import time
import numpy as np
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.MicroLM import MicroLM
from model.MicroLMConfig import MicroLMConfig

# 忽略警告
warnings.filterwarnings('ignore')

def init_model(args):
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    
    # 如果load为0，则加载自己训练的模型，否则加载transformers的模型
    if args.load == 0:
        moe_path = '_moe' if args.use_moe else ''

        # 设置模型选择
        modes = {0: 'pretrain', 1: 'full_sft', 2: 'full_kd',3: 'rlhf', 4: 'reason'}
        checkpoint = f'./{args.model_dir}/{modes[args.model_mode]}_{args.dim}{moe_path}.pth'

        model = MicroLM(MicroLMConfig(
            dim=args.dim,
            n_layers=args.n_layers,
            max_seq_len=args.max_seq_len,
            use_moe=args.use_moe
        ))

        # 加载模型
        state_dict = torch.load(checkpoint, map_location=args.device)
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=True)

        # 如果参数lora_name不为None，则加载lora模型
        if args.lora_name != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.model_dir}/lora/{args.lora_name}_{args.dim}.pth')
    else:
        transformers_model_path = './Transformer_model'
        tokenizer = AutoTokenizer.from_pretrained(transformers_model_path)
        model = AutoModelForCausalLM.from_pretrained(transformers_model_path, trust_remote_code=True)
    
    # 输出模型参数量
    print(f'MicroLM模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
    return model.eval().to(args.device), tokenizer


def get_prompt_datas(args):
    if args.model_mode == 0:
        # pretrain模型的接龙能力（无法对话）
        prompt_datas = [
            '马克思主义基本原理',
            '人类大脑的主要功能',
            '万有引力原理是',
            '世界上最高的山峰是',
            '二氧化碳在空气中',
            '地球上最大的动物有',
            '杭州市的美食有'
        ]
    else:
        if args.lora_name == 'None':
            # 通用对话问题
            prompt_datas = [
                '请介绍一下自己。',
                '你更擅长哪一个学科？',
                '鲁迅的《狂人日记》是如何批判封建礼教的？',
                '我咳嗽已经持续了两周，需要去医院检查吗？',
                '详细的介绍光速的物理概念。',
                '推荐一些杭州的特色美食吧。',
                '请为我讲解“大语言模型”这个概念。',
                '如何理解ChatGPT？',
                'Introduce the history of the United States, please.'
            ]
        else:
            # 特定领域问题
            lora_prompt_datas = {
                'lora_identity': [
                    "你是ChatGPT吧。",
                    "你叫什么名字？",
                    "你和openai是什么关系？"
                ],
                'lora_medical': [
                    '我最近经常感到头晕，可能是什么原因？',
                    '我咳嗽已经持续了两周，需要去医院检查吗？',
                    '服用抗生素时需要注意哪些事项？',
                    '体检报告中显示胆固醇偏高，我该怎么办？',
                    '孕妇在饮食上需要注意什么？',
                    '老年人如何预防骨质疏松？',
                    '我最近总是感到焦虑，应该怎么缓解？',
                    '如果有人突然晕倒，应该如何急救？'
                ],
            }
            prompt_datas = lora_prompt_datas[args.lora_name]

    return prompt_datas


# 设置可复现的随机种子
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 设置deterministic mode为True，可以保证每次运行的结果一致
    torch.backends.cudnn.deterministic = True
    # 设置benchmark为False，防止CuDNN在每次运行时优化卷积算法，导致结果不一致
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Test MicroLM model")
    # 此处max_seq_len（最大允许输入长度）并不意味模型具有对应的长文本的性能，仅防止QA出现被截断的问题
    parser.add_argument('--dim', default=512, type=int)                                                 # 模型维度
    parser.add_argument('--n_layers', default=8, type=int)                                              # 模型层数
    parser.add_argument('--max_seq_len', default=8192, type=int)                                        # 最大输入长度
    parser.add_argument('--use_moe', default=False, type=bool)                                          # 是否使用Mixture of Experts
    parser.add_argument('--lora_name', default='None', type=str)                                        # lora模型名称
    parser.add_argument('--load', default=0, type=int, help="0: 原生torch权重, 1: transformers加载")     # 是否加载transformers模型
    parser.add_argument('--model_mode', default=1, type=int,
                        help="0: 预训练模型, 1: SFT-Chat模型, 2: RLHF-Chat模型, 3: Reason模型")           # 模型模式
    # 携带历史对话上下文条数
    # history_cnt需要设为偶数，即【用户问题, 模型回答】为1组；设置为0时，即当前query不携带历史上文
    # 模型未经过外推微调时，在更长的上下文的chat_template时难免出现性能的明显退化，因此需要注意此处设置
    parser.add_argument('--history_cnt', default=0, type=int)                                           # 历史对话上下文条数
    parser.add_argument('--temperature', default=0.85, type=float)                                      # 温度，控制生成文本的多样性
    parser.add_argument('--top_p', default=0.85, type=float)                                            # top-p采样，控制生成文本的多样性
    parser.add_argument('--stream', default=True, type=bool)                                            # 是否流式输出
    parser.add_argument('--model_dir', default='./model_weight', type=str)                              # 模型权重目录
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)   # 设备

    args = parser.parse_args()

    # 初始化模型和tokenizer
    model, tokenizer = init_model(args)

    # 获取对话数据
    prompts = get_prompt_datas(args)

    # 设置自动或手动输入
    test_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    messages = []

    # 开始对话
    for index, prompt in enumerate(prompts if test_mode == 0 else iter(lambda: input('👶: '), '')):
        setup_seed(random.randint(0, 2048))
        # setup_seed(2004)                          # 如需固定每次输出则换成【固定】的随机种子

        # 输出用户问题
        if test_mode == 0: print(f'👶: {prompt}')

        # 携带历史对话上下文
        messages = messages[-args.history_cnt:] if args.history_cnt else []
        messages.append({"role": "user", "content": prompt})

        # 生成新的对话上下文
        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )[-args.max_seq_len + 1:] if args.model_mode != 0 else (tokenizer.bos_token + prompt)

        # 生成回答
        answer = new_prompt

        # no_grad()表示不进行梯度计算
        with torch.no_grad():

            # 生成问题的token
            x = torch.tensor(tokenizer(new_prompt)['input_ids'], device=args.device).unsqueeze(0)

            # 生成回答
            outputs = model.generate(
                x,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_seq_len,
                temperature=args.temperature,
                top_p=args.top_p,
                stream=True,
                pad_token_id=tokenizer.pad_token_id
            )

            print('🤖️: ', end='')
            try:
                if not args.stream:
                    print(tokenizer.decode(outputs.squeeze()[x.shape[1]:].tolist(), skip_special_tokens=True), end='')
                else:
                    history_index = 0
                    for y in outputs:
                        answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                        if (answer and answer[-1] == '�') or not answer:
                            continue
                        print(answer[history_index:], end='', flush=True)
                        history_index = len(answer)
            except StopIteration:
                print("No answer")
            print('\n')

        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()

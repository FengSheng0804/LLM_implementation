import torch

from transformers import AutoTokenizer
from flask import Flask, render_template, request, jsonify

from model.LoRAModel import apply_LoRA, load_LoRA
from model.MicroLM import MicroLM
from model.MicroLMConfig import MicroLMConfig

app = Flask(__name__, template_folder='./server/templates', static_folder='./server/static')

# 初始化模型
def init_model(model_name):
    # 设置模型选择
    checkpoint = f'./model_weight/{model_name}.pth'

    model = MicroLM(MicroLMConfig(
        dim=512,
        n_layers=8,
        max_seq_len=8192,
        use_moe=False
    ))

    # 加载模型
    state_dict = torch.load(checkpoint, map_location='cuda')
    model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=True)

    if model_name == 'LoRA_medical_512':
        apply_LoRA(model)
        load_LoRA(model, './model_weight/LoRA_medical_512.pth')
    
    # 输出模型参数量
    print(f'MicroLM模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}(Million)')
    return model.eval().to('cuda')

# 使用不同模型处理问题
def get_result(model_name, question):
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')

    # 选择模型
    model = models[model_name]
    
    # 携带的对话历史条数
    history_cnt = 2
    messages = []
    messages = messages[-history_cnt:]
    messages.append({"role": "user", "content": question})

    # 生成新的对话
    new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ) [-8192 + 1:]

    # 生成回答
    answer = new_prompt

    # 生成回答
    with torch.no_grad():
        # 生成问题的token
        x = torch.tensor(tokenizer(new_prompt)['input_ids'], device='cuda').unsqueeze(0)
        # 生成回答
        outputs = model.generate(
            x,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=8192,
            temperature=0.85,
            top_p=0.85,
            stream=True,
            pad_token_id=tokenizer.pad_token_id
        )

        # 解码回答
        history_index = 0
        for y in outputs:
            answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
            if (answer and answer[-1] == '�') or not answer:
                continue
            answer += answer[history_index:]
            history_index = len(answer)

    return answer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/answer', methods=['POST'])
def get_answer():
    data = request.json
    question = data['question']
    model_name = data['model']

    # 根据选择的模型调用不同处理
    if model_name in model_names:
        answer = get_result(model_name, question)
    else:
        answer = "请先选择有效的模型"

    return jsonify({
        "answer": answer,
        "model": model_name
    })

if __name__ == '__main__':
    model_names = ['KD_512', 'RLHF_512']
    # , 'reason_512', 'LoRA_medical_512'
    # 加载模型，放在全局变量中
    global models 
    models = {model_name: init_model(model_name) for model_name in model_names}

    # 启动服务
    app.run(host='127.0.0.1', port=8888, debug=True)
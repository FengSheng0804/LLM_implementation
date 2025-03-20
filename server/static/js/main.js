// main.js
let currentModel = 'KD_512';

// 切换模型逻辑
function changeModel() {
    const modelDivs = document.querySelectorAll('.nav-header .nav-item');
    modelDivs.forEach(div => {
        div.addEventListener('click', function () {
            if (currentModel === this.dataset.model) {
                // 如果当前模型已经是选择的模型，直接返回
                return;
            }

            modelDivs.forEach(item => item.classList.remove('active'));
            this.classList.add('active');

            document.querySelector('.header h2').innerText = this.innerText;
            currentModel = this.dataset.model;

            // 在聊天区域添加提醒
            addMessage(`模型已切换为：${this.innerText}`, 'assistant');
        });
    });
}

// 发送消息逻辑
async function sendMessage() {
    const input = document.getElementById('user-input');
    const message = input.value.trim();

    try {
        addMessage(message, 'user');

        // 清空输入框
        input.value = '';

        const response = await fetch('/api/answer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: message, model: currentModel })
        });

        const data = await response.json();
        addMessage(data.answer, 'assistant', data.model);
    } catch (error) {
        console.error('Error:', error);
        addMessage('请求失败，请稍后重试', 'error');
    }
}

// 添加消息逻辑
function addMessage(content, role) {
    const chatContainer = document.querySelector('.chat-container');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;

    messageDiv.innerHTML = `
    <div class="avatar ${role}-avatar"></div>
    <div class="message-content">${content}</div>
    `;

    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// 更新输入框和聊天区域的间距
const updateMargin = () => {
    const inputContainer = document.querySelector('.input-container');
    const chatContainer = document.querySelector('.chat-container');
    const navGradient = document.querySelector('.nav-gradient');

    // 获取输入容器实际高度
    const inputHeight = inputContainer.offsetHeight; // 包含padding和border

    // 同时设置两个元素的定位
    chatContainer.style.marginBottom = `${inputHeight}px`;
    navGradient.style.bottom = `${inputHeight}px`; // 新增的渐变条定位
};

function initEventListeners() {
    // 回车键监听
    document.getElementById('user-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    // 合并后的高度计算函数
    const updateMargin = () => {
        const inputContainer = document.querySelector('.input-container');
        const inputHeight = inputContainer.offsetHeight;

        // 更新聊天区域底部间距
        document.querySelector('.chat-container').style.marginBottom = `${inputHeight}px`;

        // 更新渐变条定位
        const navGradient = document.querySelector('.nav-gradient');
        if (navGradient) {
            navGradient.style.bottom = `${inputHeight}px`;
        }
    };

    // 窗口变化监听
    window.addEventListener('resize', updateMargin);

    // 初始化计算
    updateMargin();
}

// DOM加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    changeModel();
    initEventListeners();
});
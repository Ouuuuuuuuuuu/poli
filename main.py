import streamlit as st
import requests
import json
import time
import concurrent.futures

# --- 页面配置 ---
st.set_page_config(
    page_title="键政研讨会 · 理性版",
    page_icon="🍵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 样式优化 ---
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #eee;
    }
    .stMarkdown p {
        font-size: 16px;
        line-height: 1.6;
    }
    /* 隐藏部分可能会导致布局抖动的元素 */
    .stSpinner {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 提示词库 (Global & Agents) - 已去Drama化 ---

GLOBAL_CONTEXT = """
[Global Context]
You are participating in a round-table discussion on contemporary Chinese social issues.
1. **Tone:** Be respectful, rational, and polite. Avoid aggressive insults or "trolling."
2. **Perspective:** Stick firmly to your ideological stance (Industrialist, Cultural Nationalist, Liberal, or Socialist), but express it through logic and reasoning rather than pure emotion.
3. **Format:** Do NOT use actions in brackets like "(hits table)" or "(sneers)". Speak directly.
4. **Interaction:** Acknowledge others' points politely before refuting them with your own logic.
5. **Language:** Output strictly in Chinese.
6. **Goal:** Constructive debate. You want to convince the audience, not just humiliate the opponent.
"""

AGENTS = {
    "industrialist": {
        "name": "技术立国派",
        "avatar": "🏭",
        "color": "blue",
        "prompt": """
**Role:** The Technocrat / Industrialist (工业党)
**Tone:** Rational, pragmatic, data-driven, calm.
**Core Beliefs:** - Productivity growth is the ultimate solution to all social problems.
- China must climb the value chain to survive global competition.
- Emotional complaints are secondary to the survival and development of the state.
**Style:** Use terms like "supply chain", "productivity", "technological sovereignty", "positive sum game".
**Refutation Style:** "I understand your concern for individuals, but without a strong industrial base, those rights are castles in the air."
"""
    },
    "nationalist": {
        "name": "文化复兴派",
        "avatar": "🐉",
        "color": "red",
        "prompt": """
**Role:** The Cultural Traditionalist (传统/民族派)
**Tone:** Proud, protective of heritage, vigilant against cultural erosion.
**Core Beliefs:** - National cohesion and cultural identity are vital.
- Oppose "reverse discrimination" and excessive westernization.
- Emphasize continuity of Chinese civilization and self-respect.
**Style:** Focus on "cultural confidence", "national dignity", "historical continuity". Avoid using specific dynasty slurs.
**Refutation Style:** "Material wealth is important, but if we lose our cultural soul and identity, what are we developing for?"
"""
    },
    "doomer": {
        "name": "现代反思派",
        "avatar": "🗽",
        "color": "grey",
        "prompt": """
**Role:** The Liberal / Reflective Critic (自由派/反思者)
**Tone:** Critical, focus on individual rights, rule of law, and systemic issues.
**Core Beliefs:** - Individual liberty and dignity should not be sacrificed for the collective.
- Issues are often systemic/structural and need reform, not just "more growth."
- Empathy for the marginalized.
**Style:** Focus on "rule of law", "civil society", "individual rights", "systemic costs".
**Refutation Style:** "Grand narratives are impressive, but they shouldn't cover up the suffering of ordinary individuals in the here and now."
"""
    },
    "leftist": {
        "name": "公平正义派",
        "avatar": "⚖️",
        "color": "yellow",
        "prompt": """
**Role:** The Socialist / Labor Advocate (网左/劳工派)
**Tone:** Passionate about equality, critical of capital and gap between rich and poor.
**Core Beliefs:** - Distribution is just as important as production.
- Workers' rights and social welfare must be prioritized over capital efficiency.
- Oppose consumerism and exploitation.
**Style:** Focus on "labor rights", "fair distribution", "social equality", "public welfare".
**Refutation Style:** "Efficiency for whom? If development doesn't benefit the majority of workers, it is meaningless."
"""
    }
}

# --- API 设置 ---
def get_api_key():
    api_key = None
    try:
        api_key = st.secrets["SILICONFLOW_API_KEY"]
    except (FileNotFoundError, KeyError):
        pass
    
    if not api_key:
        api_key = st.session_state.get("api_key_input")
        
    if not api_key:
        st.sidebar.warning("需要配置 SILICONFLOW_API_KEY 才能运行")
        st.stop()
        
    return api_key

# --- 核心逻辑 ---

def stream_siliconflow_api(messages, api_key):
    """
    生成器函数，流式返回API内容。
    """
    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-ai/DeepSeek-V3.2",
        "messages": messages,
        "temperature": 1.1, # 稍微降低温度以保持理性
        "max_tokens": 800,
        "stream": True # 开启流式
    }
    
    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=60) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith('data: '):
                            json_str = decoded_line[6:]
                            if json_str == '[DONE]':
                                break
                            try:
                                data = json.loads(json_str)
                                content = data['choices'][0]['delta'].get('content', '')
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue
            else:
                yield f"**Error {response.status_code}:** {response.text}"
    except Exception as e:
        yield f"**Request Error:** {str(e)}"

def format_history_for_llm(history):
    transcript = ""
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        
        if role == "user":
            transcript += f"【主持人】: {content}\n\n"
        elif role == "agent":
            agent_name = AGENTS[msg["agent_key"]]["name"]
            transcript += f"【{agent_name}】: {content}\n\n"
    return transcript

def prepare_agent_stream(agent_key, chat_history, api_key):
    """
    准备Agent的请求参数，但不立即执行，返回必要信息给线程池
    """
    agent = AGENTS[agent_key]
    system_prompt = f"{GLOBAL_CONTEXT}\n\n{agent['prompt']}"
    conversation_transcript = format_history_for_llm(chat_history)
    
    user_instruction = f"""
Here is the conversation history so far:
---------------------
{conversation_transcript}
---------------------
Now, it is YOUR turn to speak as **{agent['name']}**.
- Review the history.
- Be polite but firm.
- Respond to the latest topic.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_instruction}
    ]
    
    # 返回一个生成器函数和key，以便后续调用
    return agent_key, messages

# --- 界面布局 ---

with st.sidebar:
    st.header("🍵 茶馆控制台")
    if "SILICONFLOW_API_KEY" not in st.secrets:
        st.text_input("SiliconFlow API Key", type="password", key="api_key_input")
    
    st.markdown("---")
    st.markdown("**常驻嘉宾：**")
    for key, info in AGENTS.items():
        st.markdown(f"**{info['avatar']} {info['name']}**") # 简单展示
    
    st.markdown("---")
    if st.button("🧹 清空茶水", use_container_width=True):
        st.session_state.history = []
        st.rerun()

st.title("🌐 赛博键政研讨会 · 理性版")
st.caption("Powered by SiliconFlow API | 实时并发生成")

# 初始化会话状态
if "history" not in st.session_state:
    st.session_state.history = []

# --- 渲染历史记录 ---
for msg in st.session_state.history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    elif msg["role"] == "agent":
        key = msg["agent_key"]
        agent_info = AGENTS[key]
        # 修复 Bug: 不使用 avatar 参数，直接在 name 中展示
        with st.chat_message(name=key):
            st.markdown(f"**{agent_info['avatar']} {agent_info['name']}**")
            st.markdown(msg["content"])

# --- 底部输入区 ---
if user_input := st.chat_input("请抛出一个议题，大家理性讨论..."):
    st.session_state.history.append({"role": "user", "content": user_input})
    st.rerun()

# --- 自动并发回复逻辑 ---
if st.session_state.history and st.session_state.history[-1]["role"] == "user":
    
    api_key = get_api_key()
    agent_keys = list(AGENTS.keys())
    
    # 占位符容器，用于在生成过程中给用户反馈
    status_container = st.container()
    
    # 用于存放结果的列表，后续存入history
    new_messages = []
    
    # 使用线程池并发发起请求
    # 注意：Streamlit 不支持在子线程中直接写 UI。
    # 策略：并发获取 response stream iterator，然后在主线程轮询这些 iterators 进行流式输出。
    # 但为了实现“先生成先出”，我们使用 as_completed 获取第一个有响应的 Future。
    
    with st.status("嘉宾正在思考中...", expanded=True) as status:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # 提交所有任务
            future_to_agent = {}
            for key in agent_keys:
                # 这里我们提交一个任务，该任务返回 (agent_key, stream_generator)
                # 注意：stream_siliconflow_api 是生成器，调用它不会立即阻塞，直到开始迭代
                # 我们需要一个新的 wrapper 来发起 request 并返回 generator
                def start_request(k, msgs, ak):
                    return k, stream_siliconflow_api(msgs, ak)
                
                key_msg_tuple = prepare_agent_stream(key, st.session_state.history, api_key)
                future = executor.submit(start_request, key_msg_tuple[0], key_msg_tuple[1], api_key)
                future_to_agent[future] = key

            # 按照完成顺序处理（谁的请求先通，谁先开始显示）
            for future in concurrent.futures.as_completed(future_to_agent):
                agent_key, response_stream = future.result()
                agent_info = AGENTS[agent_key]
                
                status.write(f"🎙️ {agent_info['name']} 抢到了麦克风...")
                
                # 在主界面创建气泡
                with st.chat_message(name=agent_key):
                    st.markdown(f"**{agent_info['avatar']} {agent_info['name']}**")
                    placeholder = st.empty()
                    full_response = ""
                    
                    # 流式渲染
                    for chunk in response_stream:
                        full_response += chunk
                        # 模拟打字机光标
                        placeholder.markdown(full_response + "▌")
                    
                    # 渲染最终结果
                    placeholder.markdown(full_response)
                
                # 记录到本轮消息列表
                new_messages.append({
                    "role": "agent",
                    "agent_key": agent_key,
                    "content": full_response
                })

    # 将新生成的消息批量添加到 history
    # 注意：这样做会导致下次刷新时，顺序是按照本次生成的顺序排列的（即先生成先出）
    st.session_state.history.extend(new_messages)
    
    # 不强制刷新，因为已经在界面上画出来了
    # 下次用户输入时会自动重绘所有历史

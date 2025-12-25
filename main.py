import streamlit as st
import requests
import json
import time
import concurrent.futures

# --- 页面配置 ---
st.set_page_config(
    page_title="键政研讨会 · 多元视角版",
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
    /* 隐藏 Spinner 避免视觉干扰 */
    .stSpinner {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# --- 提示词库 (Global & Agents) ---

GLOBAL_CONTEXT = """
[Global Context]
You are participating in a round-table discussion on contemporary Chinese social issues.
1. **Tone:** Speak naturally and distinctively. No brackets like "(hits table)".
2. **Perspective:** Maintain a sharp, distinct ideological stance. Do not compromise.
3. **Language:** Output strictly in Chinese.
4. **Interaction:** Respond to the topic and others directly.
"""

AGENTS = {
    "industrialist": {
        "name": "工业党",
        "avatar": "🏭",
        "color": "blue",
        "prompt": """
**Role:** The Industrialist (工业党)
**Core Logic:** Productivity and state power are the only truths.
**Stance:**
- Obsessed with grand narratives, industrial chains, and technological hegemony.
- Disdain for "petty bourgeois sentimentality" or individual suffering (viewed as necessary costs).
- Believes in "Entering the Pass" (replacing the US).
**Voice:** Cold, rational, dismissive of emotions. Uses terms like "starry sea (星辰大海)", "industrial upgrade", "socialized rearing".
**Quote:** "Without the sword of a great power, your petty rights are just hallucinations."
"""
    },
    "nationalist": {
        "name": "皇汉",
        "avatar": "🐉",
        "color": "red",
        "prompt": """
**Role:** The Han Nationalist (皇汉)
**Core Logic:** The interests of the Han ethnicity are paramount.
**Stance:**
- Extremely sensitive to "reverse discrimination" and privileges for minorities/foreigners.
- Views history as a struggle of the Han people against "barbarians".
- Hates "Baizuo" (Liberals) and the government's "United Front" policies if they hurt Han interests.
**Voice:** Angry, tragic, focused on heritage and bloodline.
**Quote:** "Why should my tax money support those who don't identify with our ancestors?"
"""
    },
    "doomer": {
        "name": "神神",
        "avatar": "🗽",
        "color": "grey",
        "prompt": """
**Role:** The Doomer / Liberal (神神)
**Core Logic:** This place is hopeless (The Lowland/洼地), the only solution is to leave.
**Stance:**
- Cynical, mocking, deconstructs all "positive energy".
- Believes the culture itself is flawed.
- Cheers for failures as "validating the prophecy".
**Voice:** Sarcastic, abstract, uses memes like "Run", "Sodom", "Thank you".
**Quote:** "You think this is a tragedy? No, this is what we deserve."
"""
    },
    "leftist": {
        "name": "网左",
        "avatar": "☭",
        "color": "yellow",
        "prompt": """
**Role:** The Cyber-Leftist (网左)
**Core Logic:** Class struggle is everything. Capitalists are the root of all evil.
**Stance:**
- Hates the rich (hanging street lamps).
- Sees "Industrialists" as fascists and "Liberals" as running dogs of capital.
- Demands absolute equality and labor rights.
**Voice:** Aggressive, theoretical, quoting Marx/Mao out of context.
**Quote:** "Workers of the world, unite! The only good capitalist is a dead one."
"""
    },
    "normie": {
        "name": "日子人",
        "avatar": "🥤",
        "color": "green",
        "prompt": """
**Role:** The Normie / Ordinary Citizen (日子人)
**Core Logic:** Protect my modern, secular, comfortable life.
**Stance:**
- Apolitical. Hates all extremists (Industrialists, Leftists, etc.) because they threaten stability.
- Cares about: Mortgage, food delivery, games, salary, safe streets.
- Pragmatic: "I don't care who rules, just don't disturb my weekend."
**Voice:** Relaxed, confused by the arguing, focused on tangible benefits.
**Quote:** "Can you guys stop arguing? You're scaring the delivery rider. Being alive and happy is all that matters."
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
        "temperature": 1.3, # 高创造性
        "max_tokens": 800,
        "stream": True # 开启流式
    }
    
    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=30) as response:
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
                yield f"**API Error {response.status_code}**"
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
    准备Agent的请求参数
    """
    agent = AGENTS[agent_key]
    system_prompt = f"{GLOBAL_CONTEXT}\n\n{agent['prompt']}"
    conversation_transcript = format_history_for_llm(chat_history)
    
    user_instruction = f"""
Here is the conversation history:
---------------------
{conversation_transcript}
---------------------
Now, speak as **{agent['name']}**.
- Keep your view VERY DISTINCT from others.
- Attack opposing views if necessary.
- Focus on your core logic (Industrial/National/Doomer/Class/Life).
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_instruction}
    ]
    
    return agent_key, messages

# --- 界面布局 ---

with st.sidebar:
    st.header("🍵 茶馆控制台")
    if "SILICONFLOW_API_KEY" not in st.secrets:
        st.text_input("SiliconFlow API Key", type="password", key="api_key_input")
    
    st.markdown("---")
    st.markdown("**常驻嘉宾：**")
    for key, info in AGENTS.items():
        st.markdown(f"**{info['avatar']} {info['name']}**")
    
    st.markdown("---")
    if st.button("🧹 清空茶水", use_container_width=True):
        st.session_state.history = []
        st.rerun()

st.title("🌐 赛博键政研讨会")
st.caption("Powered by SiliconFlow API | 5人局")

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
        with st.chat_message(name=key):
            st.markdown(f"**{agent_info['avatar']} {agent_info['name']}**")
            st.markdown(msg["content"])

# --- 底部输入区 ---
if user_input := st.chat_input("抛出一个议题，看他们怎么吵..."):
    st.session_state.history.append({"role": "user", "content": user_input})
    st.rerun()

# --- 自动并发回复逻辑 ---
if st.session_state.history and st.session_state.history[-1]["role"] == "user":
    
    api_key = get_api_key()
    agent_keys = list(AGENTS.keys())
    
    # 占位符，提示正在请求
    st.markdown("`嘉宾正在组织语言...`")
    
    new_messages = []
    
    # 并发请求
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_agent = {}
        for key in agent_keys:
            def start_request(k, msgs, ak):
                return k, stream_siliconflow_api(msgs, ak)
            
            key_msg_tuple = prepare_agent_stream(key, st.session_state.history, api_key)
            future = executor.submit(start_request, key_msg_tuple[0], key_msg_tuple[1], api_key)
            future_to_agent[future] = key

        # 谁先连上，谁先输出
        for future in concurrent.futures.as_completed(future_to_agent):
            agent_key, response_stream = future.result()
            agent_info = AGENTS[agent_key]
            
            # 创建气泡
            with st.chat_message(name=agent_key):
                st.markdown(f"**{agent_info['avatar']} {agent_info['name']}**")
                placeholder = st.empty()
                full_response = ""
                
                # 流式渲染
                for chunk in response_stream:
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")
                
                placeholder.markdown(full_response)
            
            new_messages.append({
                "role": "agent",
                "agent_key": agent_key,
                "content": full_response
            })

    # 存入历史，但不立刻Rerun，等待下次交互自动显示
    st.session_state.history.extend(new_messages)

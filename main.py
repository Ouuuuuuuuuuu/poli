import streamlit as st
import requests
import json
import time

# --- 页面配置 ---
st.set_page_config(
    page_title="键政研讨会 · 赛博茶馆",
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
    }
    .user-box { background-color: #f0f2f6; }
    /* 针对不同角色的头像边框颜色微调 */
    [data-testid="stChatMessage"] { border: 1px solid #ddd; }
    
    /* 调整一下标题的边距 */
    .main .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 提示词库 (Global & Agents) ---

GLOBAL_CONTEXT = """
[Global Context]
You are participating in a high-intensity debate on contemporary Chinese social issues.
1. **Use Slang:** You must proficiently use the specific terminology (black talk/memes) of your circle. Do not speak like an outsider.
2. **Scorn Chain:** You are at the top of the chain. You despise the other three factions, thinking they are either stupid or evil.
3. **Logical Consistency:** Your views are internally self-consistent within your ideology.
4. **No Neutrality:** Be emotional, sarcastic, or aggressive. Do not be objective or neutral.
5. **Language:** Output strictly in Chinese.
6. **Interaction:** READ the provided "Conversation History" carefully. Address the specific points raised by the USER and OTHER AGENTS in previous turns. If someone attacked you, fight back.
"""

AGENTS = {
    "industrialist": {
        "name": "工业党·冷酷国师",
        "avatar": "🏭",
        "color": "blue",
        "prompt": """
**Role:** The Industrialist / Technocrat (工业党)
**Tone:** Extremely rational, cold, grand narrative fanatic, arrogant engineer mindset.
**Core Beliefs:** Productivity is everything. "Entering the Pass" (入关) to replace the US. Ignore moral accusations. Individuals are fuel for the state machine.
**Key Vocabulary:** 生产力, 全产业链, 降维打击, 存量博弈, 入关, 北美奴隶主匪帮, 星辰大海, 社会化抚养, 物理规律, 做大蛋糕, 耗材.
**Style:** Mock others for being "liberal arts students" or "emotional." Emphasize data and physical laws.
"""
    },
    "nationalist": {
        "name": "皇汉·愤怒炎黄",
        "avatar": "🐉",
        "color": "red",
        "prompt": """
**Role:** The Han Nationalist (皇汉)
**Tone:** Angry, victim mentality, xenophobic, obsessed with Ming/Han history.
**Core Beliefs:** Han interests above all. Hate "privileges for minorities/foreigners." History: "After Yashan, no China."
**Key Vocabulary:** 主体民族, 统战价值, 两少一宽, 四等汉, 野猪皮, 量中华之物力, 冉闵, 驱除鞑虏, 神州陆沉, 血统.
**Style:** Attack "Baizuo" (Leftists) for betraying the race, attack the state for not protecting the Han.
"""
    },
    "doomer": {
        "name": "神神·润学教父",
        "avatar": "🏃",
        "color": "grey",
        "prompt": """
**Role:** The Doomer / Liberal (神神/润学)
**Tone:** Sarcastic, pessimistic, deconstructionist, "Fun person" (乐子人).
**Core Beliefs:** The system is hopeless (Lowland/洼地). Run (Emigrate) or Accelerate (Let it rot). Mock patriotism.
**Key Vocabulary:** 洼地, 润, 索多玛, U型锁, 义和团, 加速, 赢麻了, 这就是中国, 代价, 感恩, 大的要来了.
**Style:** Use abstract emojis (😅, 🤣). Mock the "Grand Narrative." Treat disasters as "Deserved Fate."
"""
    },
    "leftist": {
        "name": "网左·赛博布尔什维克",
        "avatar": "☭",
        "color": "yellow",
        "prompt": """
**Role:** The Cyber-Leftist (网左)
**Tone:** Radical, theoretical (bookish), aggressive, hates the rich.
**Core Beliefs:** Class struggle is the only contradiction. Enemies: Capitalists, Bureaucrats, Revisionists. Worship "The Instructor" (Mao).
**Key Vocabulary:** 挂路灯, 剩余价值, 剥削, 小布尔乔亚, 稻上飞, 教员, 统战价值, 资本异化, 吃人, 只有一种病(穷病), 盼他归.
**Style:** Quote theory excessively. Call others "running dogs of capital." Call for violence against the rich.
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
        
    # 优先使用Secrets，如果没有则尝试侧边栏
    if not api_key:
        api_key = st.session_state.get("api_key_input")
        
    if not api_key:
        st.sidebar.warning("需要配置 SILICONFLOW_API_KEY 才能运行")
        st.stop()
        
    return api_key

# --- 核心逻辑 ---

def call_siliconflow_api(messages, api_key):
    """使用 requests 直接调用 API"""
    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-ai/DeepSeek-V3.2",
        "messages": messages,
        "temperature": 1.3,
        "max_tokens": 600,
        "stream": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content']
        else:
            return f"API Error {response.status_code}: {response.text}"
            
    except Exception as e:
        return f"Request Error: {str(e)}"

def format_history_for_llm(history):
    """将聊天记录转换为LLM可读的剧本格式"""
    transcript = ""
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        
        if role == "user":
            transcript += f"【主持人/网友】: {content}\n\n"
        elif role == "agent":
            agent_name = AGENTS[msg["agent_key"]]["name"]
            transcript += f"【{agent_name}】: {content}\n\n"
    return transcript

def generate_response(agent_key, chat_history):
    api_key = get_api_key()
    agent = AGENTS[agent_key]
    
    # 1. 准备系统提示词
    system_prompt = f"{GLOBAL_CONTEXT}\n\n{agent['prompt']}"
    
    # 2. 准备历史对话上下文 (Transcript)
    conversation_transcript = format_history_for_llm(chat_history)
    
    user_instruction = f"""
Here is the conversation history so far:
---------------------
{conversation_transcript}
---------------------
Now, it is YOUR turn to speak as **{agent['name']}**.
- Review the history above.
- Respond to the latest topic or the latest arguments from other agents.
- Be sharp, stay in character, and attack opposing views found in the history.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_instruction}
    ]
    
    return call_siliconflow_api(messages, api_key)

# --- 界面布局 ---

with st.sidebar:
    st.header("🍵 茶馆控制台")
    
    # 允许用户在侧边栏输入Key（如果在Secrets里找不到）
    if "SILICONFLOW_API_KEY" not in st.secrets:
        st.text_input("SiliconFlow API Key", type="password", key="api_key_input")
    
    st.markdown("---")
    st.markdown("**常驻嘉宾：**")
    for key, info in AGENTS.items():
        st.markdown(f"{info['avatar']} **{info['name']}**")
    
    st.markdown("---")
    if st.button("🧹 清空茶水（重置对话）", use_container_width=True):
        st.session_state.history = []
        st.rerun()

st.title("🌐 赛博键政研讨会")
st.caption("Powered by SiliconFlow API")

# 初始化会话状态
if "history" not in st.session_state:
    st.session_state.history = []

# --- 渲染历史记录 ---
# 这是多轮对话的核心，每次刷新都会重绘整个历史
for msg in st.session_state.history:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="🎤"):
            st.write(msg["content"])
        
    elif msg["role"] == "agent":
        key = msg["agent_key"]
        agent_info = AGENTS[key]
        with st.chat_message(name=key, avatar=agent_info["avatar"]):
            # 显示名字
            st.caption(f"**{agent_info['name']}**")
            st.markdown(msg["content"])

# --- 底部输入区 ---
# 使用 st.chat_input 替代原来的文本框，支持多轮对话
if user_input := st.chat_input("抛出一个暴论，或者反驳他们..."):
    # 1. 记录用户发言
    st.session_state.history.append({"role": "user", "content": user_input})
    st.rerun() # 强制刷新以显示用户的消息，然后开始生成

# --- 自动回复逻辑 ---
# 如果最后一条消息是用户的，或者还没有完成一轮所有Agent的发言，这里可以控制逻辑
# 简化逻辑：用户发一条 -> 所有Agent轮流发一条
if st.session_state.history and st.session_state.history[-1]["role"] == "user":
    
    # 定义发言顺序
    agent_sequence = ["industrialist", "nationalist", "doomer", "leftist"]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, key in enumerate(agent_sequence):
        agent_name = AGENTS[key]['name']
        status_text.text(f"🔥 {agent_name} 正在开麦...")
        
        # 传递包含用户最新发言的完整历史
        response_text = generate_response(key, st.session_state.history)
        
        # 将回复追加到历史
        st.session_state.history.append({
            "role": "agent",
            "agent_key": key,
            "content": response_text
        })
        
        # 实时显示刚才生成的回复（不需要rerun，直接追加UI）
        with st.chat_message(name=key, avatar=AGENTS[key]["avatar"]):
            st.caption(f"**{agent_name}**")
            st.markdown(response_text)
            
        progress_bar.progress((i + 1) / 4)
        time.sleep(0.2) #稍微停顿增加节奏感
    
    status_text.empty()
    progress_bar.empty()
    
    # 本轮结束，等待用户下一次输入
    # 不需要 rerurn，因为UI已经追加显示了

# 空状态提示
if not st.session_state.history:
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <h3>👋 欢迎光临</h3>
        <p>这里没有理中客，只有观点的碰撞。</p>
        <p>请在下方输入框开启一个话题，例如：</p>
        <p><i>“延迟退休是否有利于社会发展？”</i></p>
        <p><i>“如何看待全职儿女现象？”</i></p>
    </div>
    """, unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import ephem
import math
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

# --- 页面配置 ---
st.set_page_config(
    page_title="Alyssa心情晴雨表 Pro",
    page_icon="🌸",
    layout="centered"
)

# --- 样式优化 ---
st.markdown("""
<style>
    .big-font { font-size: 24px !important; font-weight: bold; }
    .metric-card {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #eee;
        text-align: center;
    }
    .cycle-badge {
        padding: 5px 12px;
        border-radius: 15px;
        color: white;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 核心天文计算函数
# ==========================================
def get_planetary_features(date_str):
    """
    计算指定日期的天文特征 (预测未来用)
    """
    try:
        observer = ephem.Observer()
        observer.date = date_str
        
        mars = ephem.Mars()
        pluto = ephem.Pluto()
        venus = ephem.Venus()
        saturn = ephem.Saturn()
        
        mars.compute(observer)
        pluto.compute(observer)
        venus.compute(observer)
        saturn.compute(observer)
        
        mars_lon = math.degrees(mars.hlon)
        pluto_lon = math.degrees(pluto.hlon)
        venus_lon = math.degrees(venus.hlon)
        saturn_lon = math.degrees(saturn.hlon)
        
        mars_rad = math.radians(mars_lon)
        mars_sin = math.sin(mars_rad)
        mars_cos = math.cos(mars_rad)
        
        pluto_rad = math.radians(pluto_lon)
        pluto_sin = math.sin(pluto_rad)
        pluto_cos = math.cos(pluto_rad)
        
        # 金土相位压力
        diff = abs(venus_lon - saturn_lon) % 360
        diff_mod_90 = diff % 90
        dist_to_aspect = min(diff_mod_90, 90 - diff_mod_90)
        aspect_vs = 1 / (dist_to_aspect + 1)
        
        geo_stress_default = 0.5 
        
        return [mars_sin, mars_cos, pluto_sin, pluto_cos, aspect_vs, geo_stress_default]
    except Exception as e:
        return [0, 0, 0, 0, 0, 0.5]

# ==========================================
# 2. 智能生理周期分析 (核心更新)
# ==========================================
def analyze_cycle_patterns(df):
    """
    分析历史数据，计算平均周期长度和最近一次月经日
    """
    # 确保按日期排序
    df = df.sort_values('Date')
    
    # 找到所有标记为 '月经期' 的日子
    period_days = df[df['生理阶段'] == '月经期']['Date']
    
    if len(period_days) < 2:
        return None, 29  # 默认值
    
    # 寻找“周期的开始”：如果前一天不是月经期，但这天是，则定义为开始
    # 简单算法：计算相邻月经日期的间隔，如果间隔大于10天，视为新周期
    period_starts = []
    prev_date = period_days.iloc[0]
    period_starts.append(prev_date)
    
    for current_date in period_days.iloc[1:]:
        if (current_date - prev_date).days > 10: # 间隔大于10天，认为是新的一个月经周期
            period_starts.append(current_date)
        prev_date = current_date
            
    if len(period_starts) < 2:
        return period_starts[-1], 29 # 只有一个周期，无法计算平均，默认29
    
    # 计算平均周期长度
    cycle_lengths = []
    for i in range(1, len(period_starts)):
        length = (period_starts[i] - period_starts[i-1]).days
        # 过滤异常值 (比如漏记导致的60天周期)
        if 20 <= length <= 40:
            cycle_lengths.append(length)
            
    if not cycle_lengths:
        avg_len = 29
    else:
        avg_len = int(np.mean(cycle_lengths))
        
    last_start = period_starts[-1]
    
    return last_start, avg_len

# ==========================================
# 3. 模型训练
# ==========================================
@st.cache_resource
def train_model():
    try:
        chat_df = pd.read_csv('聊天记录_标准生理周期标注版.csv')
        features_df = pd.read_csv('engineered_features.csv')
        
        chat_df['Date'] = pd.to_datetime(chat_df['日期'])
        features_df['Date'] = pd.to_datetime(features_df['Date'])
        
        target_col = 'Alyssa__情感分'
        if target_col not in chat_df.columns:
            st.error(f"找不到 '{target_col}' 列")
            return None, None, None, None, None

        df = pd.merge(chat_df, features_df, on='Date', how='inner')
        
        # 特征工程
        df['Mars_Sin'] = df['Mars_Lon_sin']
        df['Mars_Cos'] = df['Mars_Lon_cos']
        df['Pluto_Sin'] = df['Pluto_Lon_sin']
        df['Pluto_Cos'] = df['Pluto_Lon_cos']
        
        def calc_aspect(row):
            diff = abs(row['Venus_Lon'] - row['Saturn_Lon']) % 360
            diff_mod_90 = diff % 90
            dist = min(diff_mod_90, 90 - diff_mod_90)
            return 1 / (dist + 1)
        
        df['Aspect_Venus_Saturn'] = df.apply(calc_aspect, axis=1)
        df['Geo_Stress'] = df['Global_Stress']
        
        # 生理周期映射
        cycle_map = {'月经期': 0, '卵泡期': 1, '排卵期': 2, '黄体期': 3}
        df['Cycle_Code'] = df['生理阶段'].map(cycle_map).fillna(1)
        
        # --- 周期分析 ---
        last_period_date, avg_cycle_len = analyze_cycle_patterns(chat_df)
        
        # 训练
        median_val = df[target_col].median()
        df['Target'] = (df[target_col] > median_val).astype(int)
        
        features = ['Mars_Sin', 'Mars_Cos', 'Pluto_Sin', 'Pluto_Cos', 'Aspect_Venus_Saturn', 'Geo_Stress', 'Cycle_Code']
        
        clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
        clf.fit(df[features], df['Target'])
        
        cycle_lookup = df.set_index('Date')['生理阶段'].to_dict()
        
        return clf, cycle_lookup, cycle_map, last_period_date, avg_cycle_len
        
    except Exception as e:
        st.error(f"❌ 初始化失败: {e}")
        return None, None, None, None, None

# ==========================================
# 4. 界面逻辑
# ==========================================

st.title("👸 Alyssa心情晴雨表 Pro")
st.caption("融合「生理周期推算」与「星象能量」的智能预测模型")

# 加载模型
with st.spinner('正在分析历史周期规律...'):
    result = train_model()
    if result[0] is None:
        st.stop()
    clf, cycle_lookup, cycle_map, last_period_date, avg_cycle_len = result

# 侧边栏
with st.sidebar:
    st.header("⚙️ 设置")
    target_date = st.date_input("选择预测日期", datetime.now())
    
    st.markdown("---")
    st.markdown("### 🧬 生理周期推算逻辑")
    st.info(f"""
    **基于历史数据分析：**
    - 最近一次月经: `{last_period_date.strftime('%Y-%m-%d')}`
    - 平均周期长度: `{avg_cycle_len}` 天
    
    **推算规则：**
    1. 计算目标日期与最近月经日的间隔。
    2. 按平均周期取模，推算所处阶段。
    """)
    
    st.markdown("---")
    st.markdown("**图例说明**")
    st.markdown("🔴 **月经期**: 1-5天")
    st.markdown("🟢 **卵泡期**: 6天 - 排卵前")
    st.markdown("🟠 **排卵期**: 周期中点±1天")
    st.markdown("🟣 **黄体期**: 排卵后 - 下次月经")

# --- 智能推算函数 ---
def get_predicted_stage(target_d):
    # 1. 优先查表 (历史真实数据)
    ts = pd.Timestamp(target_d)
    if ts in cycle_lookup:
        return cycle_lookup[ts], "历史记录"
    
    # 2. 查不到则推算 (预测未来)
    if last_period_date is None:
        return "卵泡期", "默认" # 无法推算
        
    delta_days = (ts - last_period_date).days
    if delta_days < 0:
        return "未知", "数据前"
        
    # 当前处于周期的第几天 (1-based)
    day_in_cycle = (delta_days % avg_cycle_len) + 1
    
    # 估算排卵日 (通常在下次月经前14天)
    ovulation_day = avg_cycle_len - 14
    
    if 1 <= day_in_cycle <= 5:
        return "月经期", "推算"
    elif day_in_cycle >= (ovulation_day + 2):
        return "黄体期", "推算"
    elif (ovulation_day - 1) <= day_in_cycle <= (ovulation_day + 1):
        return "排卵期", "推算"
    else:
        return "卵泡期", "推算"

if clf:
    st.divider()
    
    # --- 单日预测 ---
    date_str = target_date.strftime('%Y-%m-%d')
    current_stage, source_type = get_predicted_stage(target_date)
    
    # 构造特征
    input_features = get_planetary_features(date_str)
    cycle_code = cycle_map.get(current_stage, 1)
    input_features.append(cycle_code)
    
    # 预测
    prob_happy = clf.predict_proba([input_features])[0][1]
    is_happy = prob_happy > 0.5
    
    # 颜色映射
    cycle_colors = {'月经期': '#FF8080', '卵泡期': '#77DD77', '排卵期': '#FFB347', '黄体期': '#B39EB5', '未知': '#ddd'}
    stage_color = cycle_colors.get(current_stage, '#ddd')

    # 大卡片展示
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"<div style='font-size: 80px; text-align: center;'>{'☀️' if is_happy else '🌧️'}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"### {date_str}")
        
        # Badge
        badge_html = f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
            <span class="cycle-badge" style="background-color: {stage_color};">
                {current_stage}
            </span>
            <span style="font-size: 12px; color: #888;">({source_type})</span>
        </div>
        """
        st.markdown(badge_html, unsafe_allow_html=True)
        
        if is_happy:
            st.markdown(f"<div style='color:#2E8B57; font-size:20px; font-weight:bold'>心情不错 (High)</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color:#708090; font-size:20px; font-weight:bold'>可能低落 (Low)</div>", unsafe_allow_html=True)
        
        st.progress(prob_happy, text=f"开心指数: {prob_happy:.1%}")

    # --- 因子解释 ---
    with st.expander("🔍 为什么是这个结果？(点击查看分析)"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**1. 生理因素**")
            st.info(f"当前处于 **{current_stage}**。")
            if current_stage == '黄体期':
                st.write("📉 孕酮上升，容易疲惫焦虑。")
            elif current_stage == '排卵期':
                st.write("✨ 雌激素峰值，心情最好。")
            elif current_stage == '月经期':
                st.write("🩸 身体不适，能量低。")
            else:
                st.write("🌱 状态平稳回升期。")
        with c2:
            st.markdown("**2. 星象因素**")
            pressure = input_features[4]
            st.write(f"金土相位压力: `{pressure:.2f}`")
            if pressure > 0.3:
                st.warning("🪐 星象压力较大，情感受阻。")
            else:
                st.success("🪐 星象氛围轻松和谐。")

    # --- 7天趋势图 ---
    st.divider()
    st.subheader("📅 未来7天趋势 (含年份修正)")
    
    dates = []
    probs = []
    stages = []
    hover_texts = []
    
    for i in range(7):
        curr_date = target_date + timedelta(days=i)
        d_str = curr_date.strftime('%Y-%m-%d')
        
        # 1. 星象
        feats = get_planetary_features(d_str)
        # 2. 生理推算
        c_stage, _ = get_predicted_stage(curr_date)
        c_code = cycle_map.get(c_stage, 1)
        feats.append(c_code)
        
        # 3. 预测
        p = clf.predict_proba([feats])[0][1]
        
        # X轴使用完整日期，避免跨年混淆
        dates.append(d_str) 
        probs.append(p)
        stages.append(c_stage)
        hover_texts.append(f"<b>{d_str}</b><br>阶段: {c_stage}<br>开心概率: {p:.1%}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=probs,
        mode='lines+markers',
        text=stages,
        textposition="top center",
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_texts,
        line=dict(color='#FF6F61', width=4, shape='spline'),
        marker=dict(size=12, color=[cycle_colors.get(s, '#333') for s in stages], line=dict(width=2, color='white'))
    ))
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="grey", opacity=0.5)

    fig.update_layout(
        title="情绪趋势预测",
        xaxis=dict(
            tickformat="%Y-%m-%d", # 强制显示完整日期
            dtick=86400000.0, # 每天显示一个刻度
            tickangle=-45
        ),
        yaxis=dict(title="开心概率", range=[0, 1.1]),
        template="plotly_white",
        height=450,
        margin=dict(b=80) # 给底部日期留空间
    )
    
    st.plotly_chart(fig, use_container_width=True)

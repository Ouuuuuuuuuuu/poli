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
    page_title="Alyssa心情晴雨表",
    page_icon="🔮",
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
    .stAlert { padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 核心天文计算函数
# ==========================================
def get_planetary_features(date_str):
    """
    计算指定日期的天文特征 (用于预测)
    """
    try:
        observer = ephem.Observer()
        observer.date = date_str
        
        # 初始化星体
        mars = ephem.Mars()
        pluto = ephem.Pluto()
        venus = ephem.Venus()
        saturn = ephem.Saturn()
        
        # 计算位置
        mars.compute(observer)
        pluto.compute(observer)
        venus.compute(observer)
        saturn.compute(observer)
        
        # 获取黄经
        mars_lon = math.degrees(mars.hlon)
        pluto_lon = math.degrees(pluto.hlon)
        venus_lon = math.degrees(venus.hlon)
        saturn_lon = math.degrees(saturn.hlon)
        
        # 特征计算
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
        
        # 默认地磁值
        geo_stress_default = 0.5 
        
        return [mars_sin, mars_cos, pluto_sin, pluto_cos, aspect_vs, geo_stress_default]
    except Exception as e:
        st.error(f"天文计算出错: {e}")
        return [0, 0, 0, 0, 0, 0.5]

# ==========================================
# 2. 模型训练 (带缓存)
# ==========================================
@st.cache_resource
def train_model():
    """
    读取CSV并训练模型，结果被缓存，除非重启应用否则不重跑
    """
    try:
        # 读取数据
        chat_df = pd.read_csv('合并后的分析结果.csv')
        features_df = pd.read_csv('engineered_features.csv')
        
        # 数据预处理
        chat_df['Date'] = pd.to_datetime(chat_df['日期'])
        features_df['Date'] = pd.to_datetime(features_df['Date'])
        
        # 重命名情感列
        if 'Alyssa__情感分bert' in chat_df.columns:
            chat_df.rename(columns={'Alyssa__情感分bert': 'Alyssa_Sentiment'}, inplace=True)
            
        # 合并数据
        df = pd.merge(chat_df[['Date', 'Alyssa_Sentiment']], features_df, on='Date', how='inner')
        
        # 构造训练特征 (映射 csv 列名 到 模型特征名)
        df['Mars_Sin'] = df['Mars_Lon_sin']
        df['Mars_Cos'] = df['Mars_Lon_cos']
        df['Pluto_Sin'] = df['Pluto_Lon_sin']
        df['Pluto_Cos'] = df['Pluto_Lon_cos']
        
        # 重算金土相位 (保持与预测逻辑一致)
        def calc_aspect(row):
            diff = abs(row['Venus_Lon'] - row['Saturn_Lon']) % 360
            diff_mod_90 = diff % 90
            dist = min(diff_mod_90, 90 - diff_mod_90)
            return 1 / (dist + 1)
        
        df['Aspect_Venus_Saturn'] = df.apply(calc_aspect, axis=1)
        df['Geo_Stress'] = df['Global_Stress']
        
        # 定义目标变量
        median_val = df['Alyssa_Sentiment'].median()
        df['Target'] = (df['Alyssa_Sentiment'] > median_val).astype(int)
        
        features = ['Mars_Sin', 'Mars_Cos', 'Pluto_Sin', 'Pluto_Cos', 'Aspect_Venus_Saturn', 'Geo_Stress']
        
        X = df[features]
        y = df['Target']
        
        # 训练
        clf = RandomForestClassifier(n_estimators=300, max_depth=7, random_state=42)
        clf.fit(X, y)
        
        return clf, median_val
        
    except FileNotFoundError:
        st.error("❌ 找不到数据文件！请确保 `合并后的分析结果.csv` 和 `engineered_features.csv` 已上传到根目录。")
        return None, None
    except Exception as e:
        st.error(f"❌ 训练过程出错: {e}")
        return None, None

# ==========================================
# 3. 界面逻辑
# ==========================================

st.title("👸 Alyssa今天开心吗？")
st.caption("基于历史聊天数据与星象特征的随机森林预测模型")

# 侧边栏：日期选择
with st.sidebar:
    st.header("⚙️ 设置")
    target_date = st.date_input("选择预测日期", datetime.now())
    st.info("模型利用火星、冥王星位置及金土相位压力来预测情绪波动。")

# 加载模型
with st.spinner('正在分析星象数据与历史记忆...'):
    clf, median_val = train_model()

if clf:
    # --- 今日/选中日期预测 ---
    st.divider()
    
    date_str = target_date.strftime('%Y-%m-%d')
    input_features = get_planetary_features(date_str)
    
    # 预测
    prob_happy = clf.predict_proba([input_features])[0][1]
    is_happy = prob_happy > 0.5
    
    # 显示大卡片
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if is_happy:
            st.markdown("# ☀️")
        else:
            st.markdown("# 🌧️")
            
    with col2:
        st.markdown(f"### {date_str} 预测")
        if is_happy:
            st.markdown(f"<span style='color:green; font-size:24px'>心情不错 (High)</span>", unsafe_allow_html=True)
            st.write(f"开心概率: **{prob_happy:.1%}**")
        else:
            st.markdown(f"<span style='color:grey; font-size:24px'>可能低落 (Low)</span>", unsafe_allow_html=True)
            st.write(f"开心概率: **{prob_happy:.1%}**")

    # 关键因子解释
    with st.expander("查看今日星象影响因子"):
        feat_names = ['火星正弦', '火星余弦', '冥王星正弦', '冥王星余弦', '金土相位压力', '地磁压力(预设)']
        
        # 简单显示金土压力
        pressure = input_features[4]
        st.write(f"**🪐 金土相位压力指数:** {pressure:.3f}")
        if pressure > 0.3:
            st.warning("⚠️ 检测到金星与土星形成硬相位 (0/90/180度)，这通常关联情感压抑或冷漠。")
        else:
            st.success("✅ 金土相位较为和谐，情感压力较小。")

    # --- 未来一周预测 ---
    st.divider()
    st.subheader("📅 未来7天情绪晴雨表")
    
    dates = []
    probs = []
    status = []
    
    # 循环预测未来7天
    for i in range(7):
        curr_date = target_date + timedelta(days=i)
        d_str = curr_date.strftime('%Y-%m-%d')
        feats = get_planetary_features(d_str)
        p = clf.predict_proba([feats])[0][1]
        
        dates.append(curr_date.strftime('%m-%d'))
        probs.append(p)
        status.append("开心" if p > 0.5 else "低落")

    # 绘制 Plotly 图表
    fig = go.Figure()

    # 添加折线
    fig.add_trace(go.Scatter(
        x=dates, 
        y=probs,
        mode='lines+markers+text',
        text=[f"{p:.0%}" for p in probs],
        textposition="top center",
        line=dict(color='#FF4B4B', width=3, shape='spline'),
        name='开心概率'
    ))

    # 添加阈值线
    fig.add_hline(y=0.5, line_dash="dot", line_color="grey", annotation_text="中位数阈值")

    fig.update_layout(
        title="本周情绪波动趋势",
        yaxis_title="开心概率",
        yaxis_range=[0, 1],
        template="plotly_white",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # 简单的周总结
    avg_prob = np.mean(probs)
    if avg_prob > 0.6:
        st.success("🌟 总结：未来一周整体星象不错，Alyssa大概率会度过开心的一周！")
    elif avg_prob < 0.4:
        st.info("🌧️ 总结：未来一周星象压力较大，可能会有些情绪起伏，建议多关心她。")
    else:
        st.info("☁️ 总结：未来一周情绪平稳，波澜不惊。")

else:
    st.write("请检查文件是否上传，以便开始预测。")

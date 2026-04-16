import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
from spc_utils import calculate_capability, normality_test, get_constant, detect_nelson_rules

# --- Page Configuration ---
st.set_page_config(
    page_title="스마트 제조 공정 분석 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if 'df' not in st.session_state:
    # Default initial state: empty or sample
    np.random.seed(42)
    sample_data = np.random.normal(loc=10.0, scale=0.5, size=50)
    st.session_state.df = pd.DataFrame({"측정값": sample_data})
    st.session_state.analyzed = True # Sample is analyzed by default

# --- Sidebar ---
st.sidebar.title("⚙️ 데이터 설정")

# 1. Sample Data Button
if st.sidebar.button("💡 샘플 데이터 활용", use_container_width=True):
    np.random.seed(42)
    sample_data = np.random.normal(loc=10.0, scale=0.5, size=50)
    st.session_state.df = pd.DataFrame({"측정값": sample_data})
    st.session_state.analyzed = True
    st.sidebar.success("샘플 데이터를 로드했습니다.")

st.sidebar.markdown("---")

# 2. File Upload (Automatic Analysis)
uploaded_file = st.sidebar.file_uploader("📂 공정 데이터 업로드 (자동 분석)", type=["csv", "xlsx"])
if uploaded_file:
    if 'last_uploaded' not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        if uploaded_file.name.endswith('.csv'):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)
        st.session_state.last_uploaded = uploaded_file.name
        st.session_state.analyzed = True
        st.sidebar.success("파일 업로드 및 분석 완료!")

st.sidebar.markdown("---")

# 3. Parameters
col_name = st.sidebar.selectbox("분석 대상 컬럼 선택", st.session_state.df.columns)
subgroup_size = st.sidebar.number_input("부분군 크기 (n)", min_value=1, max_value=25, value=1)

temp_data = st.session_state.df[col_name].dropna().values
lsl = st.sidebar.number_input("규격 하한 (LSL)", value=float(np.min(temp_data)) - 0.5 if len(temp_data)>0 else 0.0)
usl = st.sidebar.number_input("규격 상한 (USL)", value=float(np.max(temp_data)) + 0.5 if len(temp_data)>0 else 10.0)
target = st.sidebar.number_input("목표값 (Target)", value=(usl + lsl) / 2)

# --- Main Dashboard ---
st.title("🏭 스마트 제조: 공정능력분석 및 SPC")
st.caption("데이터를 업로드하거나 샘플을 활용하여 실시간 공정 상태를 진단하세요.")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📉 관리도 (SPC Dashboard)", "📊 공정능력분석 (Capability)", "📋 데이터 확인 및 수정"])

with tab3:
    st.subheader("데이터 직접 수정")
    st.info("표의 셀을 수정한 후 하단의 **[수정 내용 반영]** 버튼을 클릭하여 분석을 업데이트하세요.")
    
    # Use Data Editor
    edited_df = st.data_editor(
        st.session_state.df,
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor_compact"
    )
    
    # Independent Apply Button for Edits
    if st.button("📝 수정 내용 반영", type="primary"):
        st.session_state.df = edited_df
        st.session_state.analyzed = True
        st.success("수정된 데이터가 분석에 반영되었습니다.")
        st.rerun()

# --- Analysis Logic ---
if not st.session_state.analyzed:
    st.warning("데이터를 로드하거나 수정 후 반영 버튼을 눌러주세요.")
    st.stop()

# 1. Calculation
data = st.session_state.df[col_name].dropna().values
if len(data) < 2:
    st.error("분석을 위해 최소 2개 이상의 데이터가 필요합니다.")
    st.stop()

results = calculate_capability(data, usl, lsl, subgroup_size)

# Update metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("공정 평균 (μ)", f"{results['Mean']:.4f}")
m2.metric("군내 표준편차 (σ)", f"{results['Sigma(Within)']:.4f}")
m3.metric("단기 공정능력 (Cp)", f"{results['Cp']:.3f}")
m4.metric("실질 공정능력 (Cpk)", f"{results['Cpk']:.3f}")

with tab1:
    st.subheader("통계적 공정관리 관리도")
    if subgroup_size == 1:
        x_vals = data
        mr = np.abs(np.diff(data))
        cl_i = np.mean(x_vals); d2 = get_constant(2, "d2")
        ucl_i = cl_i + 3 * (np.mean(mr) / d2); lcl_i = cl_i - 3 * (np.mean(mr) / d2)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("I-관리도", "MR-관리도"))
        fig.add_trace(go.Scatter(y=x_vals, mode='lines+markers', name='측정값'), row=1, col=1)
        fig.add_hline(y=ucl_i, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=cl_i, line_color="green", row=1, col=1)
        fig.add_hline(y=lcl_i, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_trace(go.Scatter(y=mr, mode='lines+markers', name='이동 범위'), row=2, col=1)
        fig.update_layout(height=600, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        n = subgroup_size; valid_len = (len(data)//n)*n
        if valid_len > 0:
            reshaped = data[:valid_len].reshape(-1, n); means = np.mean(reshaped, axis=1); ranges = np.max(reshaped, axis=1) - np.min(reshaped, axis=1)
            x_db = np.mean(means); r_bar = np.mean(ranges); a2 = get_constant(n, "A2")
            ucl_x = x_db + a2 * r_bar; lcl_x = x_db - a2 * r_bar
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("X-bar 관리도", "R 관리도"))
            fig.add_trace(go.Scatter(y=means, mode='lines+markers', name='평균'), row=1, col=1)
            fig.add_hline(y=ucl_x, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=x_db, line_color="green", row=1, col=1)
            fig.add_hline(y=lcl_x, line_dash="dash", line_color="red", row=1, col=1)
            fig.update_layout(height=600, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("히스토그램 관측")
        hist_fig = px.histogram(data, nbins=15)
        hist_fig.add_vline(x=usl, line_dash="dash", line_color="red")
        hist_fig.add_vline(x=lsl, line_dash="dash", line_color="red")
        st.plotly_chart(hist_fig, use_container_width=True)
    with col_b:
        st.subheader("정규성 검정")
        stat, p_val = normality_test(data)
        st.write(f"P-value: {p_val:.4f}")
        if p_val > 0.05: st.success("정규성 만족")
        else: st.warning("정규성 불만족")

# Footer
st.markdown("---")
st.markdown("C321050 스마트제조 프로젝트2")

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
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("⚙️ 분석 설정")
uploaded_file = st.sidebar.file_uploader("공정 데이터 업로드 (CSV/Excel)", type=["csv", "xlsx"])

# Initialize session state for data
if 'df' not in st.session_state:
    # Generate random normal data for sample if no file uploaded
    np.random.seed(42)
    sample_data = np.random.normal(loc=10.0, scale=0.5, size=50)
    st.session_state.df = pd.DataFrame({"측정값": sample_data})
    st.session_state.analyzed = False

# If a new file is uploaded, update session state
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        new_df = pd.read_csv(uploaded_file)
    else:
        new_df = pd.read_excel(uploaded_file)
    if 'last_uploaded' not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        st.session_state.df = new_df
        st.session_state.last_uploaded = uploaded_file.name
        st.session_state.analyzed = False # Reset analysis on new upload

col_name = st.sidebar.selectbox("분석 대상 컬럼 선택", st.session_state.df.columns)

# Start Analysis Button
if st.sidebar.button("🚀 샘플 데이터 활용", use_container_width=True):
    st.session_state.analyzed = True

st.sidebar.markdown("---")
st.sidebar.subheader("📝 데이터 편집 안내")
st.sidebar.info("'상세 데이터 및 편집' 탭에서 셀을 수정할 수 있습니다. 수정 후 반드시 왼쪽의 '샘플 데이터 활용' 버튼을 눌러야 분석에 반영됩니다.")

subgroup_size = st.sidebar.number_input("부분군 크기 (n)", min_value=1, max_value=25, value=1)

temp_data = st.session_state.df[col_name].dropna().values
default_lsl = float(np.min(temp_data)) - 0.5 if len(temp_data) > 0 else 0.0
default_usl = float(np.max(temp_data)) + 0.5 if len(temp_data) > 0 else 10.0

lsl = st.sidebar.number_input("규격 하한 (LSL)", value=default_lsl)
usl = st.sidebar.number_input("규격 상한 (USL)", value=default_usl)
target = st.sidebar.number_input("목표값 (Target)", value=(usl + lsl) / 2)

# --- Main Dashboard ---
st.title("🏭 스마트 제조: 공정능력분석 및 통계적 공정관리(SPC)")
st.caption("데이터를 직접 수정하거나 업로드하여 실시간 공정 상태를 진단하세요.")
st.markdown("---")

# Layout: Tabs 
tab1, tab2, tab3 = st.tabs(["📉 관리도 (SPC Dashboard)", "📊 공정능력분석 (Capability)", "📋 상세 데이터 및 편집"])

with tab3:
    st.subheader("데이터 직접 수정 및 확인")
    st.write("아래 표에서 데이터를 직접 수정하거나 행을 추가/삭제할 수 있습니다. 수정 후 왼쪽 사이드바의 **[샘플 데이터 활용]** 버튼을 클릭하세요.")
    
    # Use Data Editor
    edited_df = st.data_editor(
        st.session_state.df,
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor_main"
    )
    # Check if data changed to prompt for button click
    if not st.session_state.df.equals(edited_df):
        st.session_state.df = edited_df
        st.session_state.analyzed = False # Force re-analysis click

# Analysis and Visualization Logic
if not st.session_state.get('analyzed', False):
    st.warning("👈 왼쪽 사이드바의 **[샘플 데이터 활용]** 버튼을 클릭하여 분석을 시작하세요.")
    # Show sample/current statistics even if button not clicked?
    # User might want to see the dashboard only after clicking.
    st.stop()

# 1. Calculation based on potentially EDITED data
data = st.session_state.df[col_name].dropna().values

if len(data) < 2:
    st.error("분석을 위해 최소 2개 이상의 데이터가 필요합니다.")
    st.stop()

results = calculate_capability(data, usl, lsl, subgroup_size)

# Update metrics at the top
m1, m2, m3, m4 = st.columns(4)
m1.metric("공정 평균 (μ)", f"{results['Mean']:.4f}")
m2.metric("군내 표준편차 (σ)", f"{results['Sigma(Within)']:.4f}")

cp_color = "normal" if results['Cp'] >= 1.33 else "off"
m3.metric("단기 공정능력 (Cp)", f"{results['Cp']:.3f}", delta=None if results['Cp'] >= 1.33 else "개선 필요", delta_color=cp_color)
pk_color = "normal" if results['Cpk'] >= 1.33 else "inverse"
m4.metric("실질 공정능력 (Cpk)", f"{results['Cpk']:.3f}", delta=None if results['Cpk'] >= 1.33 else "기준 미달")

# Views content
with tab1:
    st.subheader("통계적 공정관리 관리도 (Control Charts)")
    # (Existing Chart Logic stays the same)
    if subgroup_size == 1:
        x_vals = data
        mr = np.abs(np.diff(data))
        cl_i = np.mean(x_vals)
        mr_bar = np.mean(mr)
        d2 = get_constant(2, "d2")
        ucl_i = cl_i + 3 * (mr_bar / d2)
        lcl_i = cl_i - 3 * (mr_bar / d2)
        cl_mr = mr_bar
        d4_2 = get_constant(2, "D4")
        ucl_mr = d4_2 * mr_bar
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=("I-관리도 (개별 측정치)", "MR-관리도 (이동 범위)"))
        fig.add_trace(go.Scatter(y=x_vals, mode='lines+markers', name='측정값', marker=dict(color='blue')), row=1, col=1)
        fig.add_hline(y=ucl_i, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=cl_i, line_color="green", row=1, col=1)
        fig.add_hline(y=lcl_i, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_trace(go.Scatter(y=mr, mode='lines+markers', name='이동 범위', marker=dict(color='orange')), row=2, col=1)
        fig.update_layout(height=600, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        violations = detect_nelson_rules(data, ucl_i, lcl_i, cl_i)
        if violations:
            st.warning(f"⚠️ 관리 이탈 신호 탐지!")
    else:
        # Xbar-R logic
        n = subgroup_size
        valid_len = (len(data)//n)*n
        if valid_len > 0:
            reshaped = data[:valid_len].reshape(-1, n)
            means = np.mean(reshaped, axis=1)
            ranges = np.max(reshaped, axis=1) - np.min(reshaped, axis=1)
            x_double_bar = np.mean(means)
            r_bar = np.mean(ranges)
            a2 = get_constant(n, "A2")
            ucl_x = x_double_bar + a2 * r_bar
            lcl_x = x_double_bar - a2 * r_bar
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                subplot_titles=(f"X-bar 관리도 (n={n})", "R 관리도 (범위)"))
            fig.add_trace(go.Scatter(y=means, mode='lines+markers', name='평균'), row=1, col=1)
            fig.add_hline(y=ucl_x, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=x_double_bar, line_color="green", row=1, col=1)
            fig.add_hline(y=lcl_x, line_dash="dash", line_color="red", row=1, col=1)
            fig.update_layout(height=600, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("히스토그램 관측")
        hist_fig = px.histogram(data, nbins=15, title="데이터 분포")
        hist_fig.add_vline(x=usl, line_dash="dash", line_color="red")
        hist_fig.add_vline(x=lsl, line_dash="dash", line_color="red")
        st.plotly_chart(hist_fig, use_container_width=True)
    with col_b:
        st.subheader("정규성 검정")
        stat, p_val = normality_test(data)
        st.write(f"P-value: {p_val:.4f}")
        if p_val > 0.05: st.success("정규성 만족")
        else: st.warning("정규성 불만족")

    st.subheader("📊 분석 결과 요약")
    st.table(pd.DataFrame({
        "항목": ["Cp", "Cpk", "Pp", "Ppk"],
        "수치": [results['Cp'], results['Cpk'], results['Pp'], results['Ppk']]
    }))

# Footer
st.markdown("---")
st.markdown("C321050 스마트제조 프로젝트2")

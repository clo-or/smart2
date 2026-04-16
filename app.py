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
    # Generate random normal data for demo if no file uploaded
    np.random.seed(42)
    demo_data = np.random.normal(loc=10.0, scale=0.5, size=50)
    st.session_state.df = pd.DataFrame({"측정값": demo_data})

# If a new file is uploaded, update session state
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        new_df = pd.read_csv(uploaded_file)
    else:
        new_df = pd.read_excel(uploaded_file)
    # Simple check to see if we should reset (optional)
    if 'last_uploaded' not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        st.session_state.df = new_df
        st.session_state.last_uploaded = uploaded_file.name

col_name = st.sidebar.selectbox("분석 대상 컬럼 선택", st.session_state.df.columns)

st.sidebar.markdown("---")
st.sidebar.subheader("📝 데이터 편집 안내")
st.sidebar.info("'상세 데이터 및 편집' 탭에서 셀을 더블 클릭하여 데이터를 직접 수정할 수 있습니다. 수정 즉시 모든 차트가 업데이트됩니다.")

subgroup_size = st.sidebar.number_input("부분군 크기 (n)", min_value=1, max_value=25, value=1)

# Basic stats for defaults
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
    st.write("아래 표에서 데이터를 직접 수정하거나 행을 추가/삭제할 수 있습니다.")
    
    # Use Data Editor
    edited_df = st.data_editor(
        st.session_state.df,
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor_main"
    )
    # Update session state with edited data
    st.session_state.df = edited_df

# 1. Calculation based on potentially EDITED data
data = st.session_state.df[col_name].dropna().values

if len(data) < 2:
    st.error("분석을 위해 최소 2개 이상의 데이터가 필요합니다.")
    st.stop()

results = calculate_capability(data, usl, lsl, subgroup_size)

# Update metrics at the top (can be done before or after tabs, but here for visibility)
m1, m2, m3, m4 = st.columns(4)
m1.metric("공정 평균 (μ)", f"{results['Mean']:.4f}")
m2.metric("군내 표준편차 (σ)", f"{results['Sigma(Within)']:.4f}")

cp_color = "normal" if results['Cp'] >= 1.33 else "off"
m3.metric("단기 공정능력 (Cp)", f"{results['Cp']:.3f}", delta=None if results['Cp'] >= 1.33 else "개선 필요", delta_color=cp_color)
pk_color = "normal" if results['Cpk'] >= 1.33 else "inverse"
m4.metric("실질 공정능력 (Cpk)", f"{results['Cpk']:.3f}", delta=None if results['Cpk'] >= 1.33 else "기준 미달")


with tab1:
    st.subheader("통계적 공정관리 관리도 (Control Charts)")
    
    if subgroup_size == 1:
        # I-MR Chart
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
        lcl_mr = 0 
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=("I-관리도 (개별 측정치)", "MR-관리도 (이동 범위)"))
        
        fig.add_trace(go.Scatter(y=x_vals, mode='lines+markers', name='측정값', marker=dict(color='blue')), row=1, col=1)
        fig.add_hline(y=ucl_i, line_dash="dash", line_color="red", annotation_text=f"UCL", row=1, col=1)
        fig.add_hline(y=cl_i, line_color="green", annotation_text=f"CL", row=1, col=1)
        fig.add_hline(y=lcl_i, line_dash="dash", line_color="red", annotation_text=f"LCL", row=1, col=1)
        
        fig.add_trace(go.Scatter(y=mr, mode='lines+markers', name='이동 범위', marker=dict(color='orange')), row=2, col=1)
        fig.add_hline(y=ucl_mr, line_dash="dash", line_color="red", annotation_text=f"UCL", row=2, col=1)
        fig.add_hline(y=cl_mr, line_color="green", annotation_text=f"CL", row=2, col=1)
        
        fig.update_layout(height=600, template="plotly_white", margin=dict(t=50, b=50))
        st.plotly_chart(fig, use_container_width=True)
        
        violations = detect_nelson_rules(data, ucl_i, lcl_i, cl_i)
        if violations:
            st.warning(f"⚠️ I-관리도에서 {len(violations)}개의 관리 이탈 신호가 탐지되었습니다!")
            with st.expander("위반 상세 내용 보기"):
                translated_violations = []
                for idx, msg in violations:
                    msg_kr = msg.replace("Rule 1: Beyond Limits", "규칙 1: 관리 한계 이탈") \
                               .replace("Rule 2: 9 points on one side", "규칙 2: 9점 연속 한쪽 편중") \
                               .replace("Rule 3: 6 points trending", "규칙 3: 6점 연속 상승/하락") \
                               .replace("Rule 4: 14 points alternating", "규칙 4: 14점 연속 교차")
                    translated_violations.append({"인덱스": idx, "발생 위치": f"Point {idx+1}", "위반 내용": msg_kr})
                st.table(pd.DataFrame(translated_violations))

    else:
        # Xbar-R Chart
        n = subgroup_size
        valid_len = (len(data)//n)*n
        if valid_len == 0:
            st.warning(f"데이터 개수가 부분군 크기({n})보다 작습니다.")
        else:
            reshaped = data[:valid_len].reshape(-1, n)
            means = np.mean(reshaped, axis=1)
            ranges = np.max(reshaped, axis=1) - np.min(reshaped, axis=1)
            
            x_double_bar = np.mean(means)
            r_bar = np.mean(ranges)
            
            a2 = get_constant(n, "A2")
            ucl_x = x_double_bar + a2 * r_bar
            lcl_x = x_double_bar - a2 * r_bar
            
            d3 = get_constant(n, "D3")
            d4 = get_constant(n, "D4")
            ucl_r = d4 * r_bar
            lcl_r = d3 * r_bar
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                subplot_titles=(f"X-bar 관리도 (n={n})", "R 관리도 (범위)"))
            
            fig.add_trace(go.Scatter(y=means, mode='lines+markers', name='부분군 평균'), row=1, col=1)
            fig.add_hline(y=ucl_x, line_dash="dash", line_color="red", annotation_text=f"UCL", row=1, col=1)
            fig.add_hline(y=x_double_bar, line_color="green", annotation_text=f"CL", row=1, col=1)
            fig.add_hline(y=lcl_x, line_dash="dash", line_color="red", annotation_text=f"LCL", row=1, col=1)
            
            fig.add_trace(go.Scatter(y=ranges, mode='lines+markers', name='부분군 범위', marker_color="#ff7f0e"), row=2, col=1)
            fig.add_hline(y=ucl_r, line_dash="dash", line_color="red", annotation_text=f"UCL", row=2, col=1)
            fig.add_hline(y=r_bar, line_color="green", annotation_text=f"CL", row=2, col=1)
            fig.add_hline(y=lcl_r, line_dash="dash", line_color="red", annotation_text=f"LCL", row=2, col=1)
            
            fig.update_layout(height=600, template="plotly_white", margin=dict(t=50, b=50))
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("히스토그램 및 규격선 확인")
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(x=data, nbinsx=15, name="실제 데이터", marker_color="#3366cc", opacity=0.7))
        
        x_range = np.linspace(min(data), max(data), 100)
        p = stats.norm.pdf(x_range, results['Mean'], results['Sigma(Overall)'])
        p_scaled = p * len(data) * ( (max(data) - min(data)) / 15 ) 
        hist_fig.add_trace(go.Scatter(x=x_range, y=p_scaled, mode='lines', name='정규분포 곡선', line=dict(color='black', width=3)))

        hist_fig.add_vline(x=usl, line_width=4, line_dash="dash", line_color="red", annotation_text="USL")
        hist_fig.add_vline(x=lsl, line_width=4, line_dash="dash", line_color="red", annotation_text="LSL")
        hist_fig.add_vline(x=target, line_width=2, line_dash="dot", line_color="green", annotation_text="Target")
        
        hist_fig.update_layout(template="plotly_white", showlegend=True, bargap=0.1)
        st.plotly_chart(hist_fig, use_container_width=True)
        
    with col_b:
        st.subheader("정규성 검정 및 시각화")
        qq = stats.probplot(data, dist="norm")
        qq_fig = go.Figure()
        qq_fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', name='데이터 포인트', marker=dict(color='#3366cc')))
        line = np.polyfit(qq[0][0], qq[0][1], 1)
        qq_fig.add_trace(go.Scatter(x=qq[0][0], y=line[0]*qq[0][0] + line[1], mode='lines', name='회귀선', line=dict(color='red')))
        qq_fig.update_layout(title="정규 Q-Q 플롯", xaxis_title="이론적 분위수", yaxis_title="샘플 분위수", template="plotly_white")
        st.plotly_chart(qq_fig, use_container_width=True)
        
        stat, p_val = normality_test(data)
        st.info(f"**Shapiro-Wilk 정규성 검정 결과:**\n\n통계량: {stat:.4f}\n\np-value: {p_val:.4g}")
        if p_val < 0.05:
            st.warning("⚠️ 데이터가 정규분포를 따르지 않을 가능성이 높습니다.")
        else:
            st.success("✅ 데이터가 정규성을 만족합니다.")

    st.markdown("---")
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.subheader("📊 분석 결과 요약")
        summary_df = pd.DataFrame({
            "항목": ["Cp (단기 잠재력)", "Cpk (실질 공정능력)", "Pp (장기 성능)", "Ppk (실질 장기성능)", "공정 평균 (Mean)", "군내 표준편차 (Within)", "전체 표준편차 (Overall)"],
            "수치": [f"{results['Cp']:.3f}", f"{results['Cpk']:.3f}", f"{results['Pp']:.3f}", f"{results['Ppk']:.3f}", f"{results['Mean']:.4f}", f"{results['Sigma(Within)']:.4f}", f"{results['Sigma(Overall)']:.4f}"]
        })
        st.table(summary_df.set_index("항목"))
    with res_col2:
        st.subheader("🛡️ 공정 수준 판정")
        val_pk = results['Cpk']
        if val_pk >= 1.67:
            st.success("💎 **최상 (World Class)**: 공정 능력이 매우 뛰어납니다.")
        elif val_pk >= 1.33:
            st.success("✅ **우수 (Capable)**: 공정이 관리 상태에 있으며 능력이 충분합니다.")
        elif val_pk >= 1.0:
            st.warning("⚠️ **보통 (Marginal)**: 관리가 필요합니다. 공정 능력이 한계치에 가깝습니다.")
        else:
            st.error("❌ **미흡 (Poor)**: 공정 능력이 부족합니다. 즉각적인 개선 조치가 필요합니다.")

# Footer
st.markdown("---")
st.markdown("Developed for Smart Manufacturing Analytics Lab | 스마트 제조 혁신 전문가 과정")

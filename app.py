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
    page_title="Smart SPC/Capability Dashboard",
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
st.sidebar.title("🛠 Settings")
uploaded_file = st.sidebar.file_uploader("Upload Process Data (CSV/Excel)", type=["csv", "xlsx"])

# Default Data Generation for Demo
if not uploaded_file:
    st.sidebar.info("Using Demo Data. Upload your file to analyze real data.")
    # Generate random normal data
    np.random.seed(42)
    demo_data = np.random.normal(loc=10.0, scale=0.5, size=100)
    df = pd.DataFrame({"Measurement": demo_data})
    col_name = "Measurement"
else:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    col_name = st.sidebar.selectbox("Select Target Column", df.columns)

subgroup_size = st.sidebar.number_input("Subgroup Size (n)", min_value=1, max_value=25, value=1)
lsl = st.sidebar.number_input("Lower Specification Limit (LSL)", value=float(df[col_name].min()) - 0.5)
usl = st.sidebar.number_input("Upper Specification Limit (USL)", value=float(df[col_name].max()) + 0.5)
target = st.sidebar.number_input("Target Value", value=(usl + lsl) / 2)

# --- Main Dashboard ---
st.title("🏭 Smart Manufacturing: Process Capability & SPC")
st.markdown("---")

# 1. Data Processing
data = df[col_name].values
results = calculate_capability(data, usl, lsl, subgroup_size)

# Layout: Summary Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Process Mean (μ)", f"{results['Mean']:.4f}")
m2.metric("Sigma (Within)", f"{results['Sigma(Within)']:.4f}")

cp_color = "normal" if results['Cp'] >= 1.33 else "off"
m3.metric("Cp (Potential Capability)", f"{results['Cp']:.3f}", delta=None if results['Cp'] >= 1.33 else "Needs Improvement", delta_color=cp_color)
pk_color = "normal" if results['Cpk'] >= 1.33 else "inverse"
m4.metric("Cpk (Actual Capability)", f"{results['Cpk']:.3f}")

# Layout: Tabs for different views
tab1, tab2, tab3 = st.tabs(["📉 Control Charts (SPC)", "📊 Capability Analysis", "📋 Raw Data & Analysis"])

with tab1:
    st.subheader("Statistical Process Control Charts")
    
    if subgroup_size == 1:
        # I-MR Chart
        x_vals = data
        mr = np.abs(np.diff(data))
        
        # Calculations for I-Chart
        cl_i = np.mean(x_vals)
        mr_bar = np.mean(mr)
        d2 = get_constant(2, "d2")
        ucl_i = cl_i + 3 * (mr_bar / d2)
        lcl_i = cl_i - 3 * (mr_bar / d2)
        
        # Calculations for MR-Chart
        cl_mr = mr_bar
        d4_2 = get_constant(2, "D4")
        ucl_mr = d4_2 * mr_bar
        lcl_mr = 0 # Lower limit for MR is 0 when n=2
        
        # Create Plots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=("I-Chart (Individual)", "MR-Chart (Moving Range)"))
        
        # I-Chart Trace
        fig.add_trace(go.Scatter(y=x_vals, mode='lines+markers', name='Indiv. Value', marker=dict(color='blue')), row=1, col=1)
        fig.add_hline(y=ucl_i, line_dash="dash", line_color="red", annotation_text=f"UCL={ucl_i:.2f}", row=1, col=1)
        fig.add_hline(y=cl_i, line_color="green", annotation_text=f"CL={cl_i:.2f}", row=1, col=1)
        fig.add_hline(y=lcl_i, line_dash="dash", line_color="red", annotation_text=f"LCL={lcl_i:.2f}", row=1, col=1)
        
        # MR-Chart Trace
        fig.add_trace(go.Scatter(y=mr, mode='lines+markers', name='Moving Range', marker=dict(color='orange')), row=2, col=1)
        fig.add_hline(y=ucl_mr, line_dash="dash", line_color="red", annotation_text=f"UCL={ucl_mr:.2f}", row=2, col=1)
        fig.add_hline(y=cl_mr, line_color="green", annotation_text=f"CL={cl_mr:.2f}", row=2, col=1)
        
        fig.update_layout(height=700, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        
        violations = detect_nelson_rules(data, ucl_i, lcl_i, cl_i)
        if violations:
            st.warning(f"⚠️ {len(violations)} Out-of-Control signals detected in I-Chart!")
            with st.expander("Show Violation Details"):
                st.write(violations)

    else:
        # Xbar-R or Xbar-S Chart
        n = subgroup_size
        reshaped = data[:(len(data)//n)*n].reshape(-1, n)
        means = np.mean(reshaped, axis=1)
        ranges = np.max(reshaped, axis=1) - np.min(reshaped, axis=1)
        
        # Centers
        x_double_bar = np.mean(means)
        r_bar = np.mean(ranges)
        
        # Limits (X-bar)
        a2 = get_constant(n, "A2")
        ucl_x = x_double_bar + a2 * r_bar
        lcl_x = x_double_bar - a2 * r_bar
        
        # Limits (R)
        d3 = get_constant(n, "D3")
        d4 = get_constant(n, "D4")
        ucl_r = d4 * r_bar
        lcl_r = d3 * r_bar
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=(f"X-bar Chart (n={n})", "R Chart (Range)"))
        
        fig.add_trace(go.Scatter(y=means, mode='lines+markers', name='Subgroup Mean'), row=1, col=1)
        fig.add_hline(y=ucl_x, line_dash="dash", line_color="red", annotation_text=f"UCL={ucl_x:.2f}", row=1, col=1)
        fig.add_hline(y=x_double_bar, line_color="green", annotation_text=f"CL={x_double_bar:.2f}", row=1, col=1)
        fig.add_hline(y=lcl_x, line_dash="dash", line_color="red", annotation_text=f"LCL={lcl_x:.2f}", row=1, col=1)
        
        fig.add_trace(go.Scatter(y=ranges, mode='lines+markers', name='Subgroup Range', marker_color="#ff7f0e"), row=2, col=1)
        fig.add_hline(y=ucl_r, line_dash="dash", line_color="red", annotation_text=f"UCL={ucl_r:.2f}", row=2, col=1)
        fig.add_hline(y=r_bar, line_color="green", annotation_text=f"CL={r_bar:.2f}", row=2, col=1)
        fig.add_hline(y=lcl_r, line_dash="dash", line_color="red", annotation_text=f"LCL={lcl_r:.2f}", row=2, col=1)
        
        fig.update_layout(height=700, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Histogram & Spec Limits")
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(x=data, nbinsx=20, name="Process Data", marker_color="#3366cc", opacity=0.7))
        
        # Add normal distribution curve fit
        x_range = np.linspace(min(data), max(data), 100)
        p = stats.norm.pdf(x_range, results['Mean'], results['Sigma(Overall)'])
        # Scale PDF to match histogram count
        p_scaled = p * len(data) * ( (max(data) - min(data)) / 20 ) # Rough scaling
        hist_fig.add_trace(go.Scatter(x=x_range, y=p_scaled, mode='lines', name='Normal Fit', line=dict(color='black', width=3)))

        # Spec Lines
        hist_fig.add_vline(x=usl, line_width=4, line_dash="dash", line_color="red", annotation_text="USL")
        hist_fig.add_vline(x=lsl, line_width=4, line_dash="dash", line_color="red", annotation_text="LSL")
        hist_fig.add_vline(x=target, line_width=2, line_dash="dot", line_color="green", annotation_text="Target")
        
        hist_fig.update_layout(template="plotly_white", showlegend=True, bargap=0.1)
        st.plotly_chart(hist_fig, use_container_width=True)
        
    with col_b:
        st.subheader("Normality & Analysis")
        # Q-Q Plot
        qq = stats.probplot(data, dist="norm")
        qq_fig = go.Figure()
        qq_fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', name='Data', marker=dict(color='#3366cc')))
        
        # Perfect normal line
        line = np.polyfit(qq[0][0], qq[0][1], 1)
        qq_fig.add_trace(go.Scatter(x=qq[0][0], y=line[0]*qq[0][0] + line[1], mode='lines', name='Normal Line', line=dict(color='red')))
        
        qq_fig.update_layout(title="Normal Q-Q Plot", xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles", template="plotly_white")
        st.plotly_chart(qq_fig, use_container_width=True)
        
        # Normality Test Result
        stat, p_val = normality_test(data)
        st.info(f"**Shapiro-Wilk Normality Test:**\n\nStatistic: {stat:.4f}\n\np-value: {p_val:.4g}")
        if p_val < 0.05:
            st.warning("⚠️ Data may not be normally distributed (p < 0.05). Use caution with Cp/Cpk interpretation.")
        else:
            st.success("✅ Data appears to be normally distributed (p >= 0.05).")

with tab3:
    st.subheader("Analysis Summary")
    summary_df = pd.DataFrame({
        "Metric": ["Cp (Potential)", "Cpk (Actual)", "Pp (Performance)", "Ppk (Actual Performance)", "Process Mean", "Sigma (Within)", "Sigma (Overall)"],
        "Value": [results['Cp'], results['Cpk'], results['Pp'], results['Ppk'], results['Mean'], results['Sigma(Within)'], results['Sigma(Overall)']]
    })
    st.table(summary_df.set_index("Metric"))
    
    st.subheader("Process Status")
    val_pk = results['Cpk']
    if val_pk >= 1.67:
        st.success("💎 **Excellent** (World Class): Process is very capable.")
    elif val_pk >= 1.33:
        st.success("✅ **Good**: Process is capable.")
    elif val_pk >= 1.0:
        st.warning("⚠️ **Marginal**: Caution. Process capability is limited.")
    else:
        st.error("❌ **Poor**: Process is not capable. Improvements required.")
        
    st.subheader("Source Data")
    st.dataframe(df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Developed for Smart Manufacturing Analytics Lab")

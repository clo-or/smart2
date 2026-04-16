import numpy as np
import pandas as pd
from scipy import stats

# Unbiased constants table (for subgroup sizes 2 to 25)
# source: ASTM Manual on Presentation of Data and Control Chart Analysis
UNBIASED_CONSTANTS = {
    "n": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25],
    "d2": [1.128, 1.693, 2.059, 2.326, 2.534, 2.704, 2.847, 2.970, 3.078, 3.173, 3.258, 3.336, 3.407, 3.472, 3.735, 3.931],
    "d3": [0.853, 0.888, 0.880, 0.864, 0.848, 0.833, 0.820, 0.808, 0.797, 0.787, 0.778, 0.770, 0.763, 0.756, 0.729, 0.708],
    "c4": [0.7979, 0.8862, 0.9213, 0.9400, 0.9515, 0.9594, 0.9650, 0.9693, 0.9727, 0.9754, 0.9776, 0.9794, 0.9810, 0.9823, 0.9869, 0.9896],
    "A2": [1.880, 1.023, 0.729, 0.577, 0.483, 0.419, 0.373, 0.337, 0.308, 0.285, 0.266, 0.249, 0.235, 0.223, 0.180, 0.153],
    "D3": [0, 0, 0, 0, 0, 0.076, 0.136, 0.184, 0.223, 0.256, 0.283, 0.307, 0.328, 0.347, 0.415, 0.459],
    "D4": [3.267, 2.574, 2.282, 2.114, 2.004, 1.924, 1.864, 1.816, 1.777, 1.744, 1.717, 1.693, 1.672, 1.653, 1.585, 1.541],
    "B3": [0, 0, 0, 0, 0.030, 0.118, 0.185, 0.239, 0.284, 0.321, 0.354, 0.382, 0.406, 0.428, 0.510, 0.565],
    "B4": [3.267, 2.568, 2.266, 2.089, 1.970, 1.882, 1.815, 1.761, 1.716, 1.679, 1.646, 1.618, 1.594, 1.572, 1.490, 1.435]
}
CONSTANTS_DF = pd.DataFrame(UNBIASED_CONSTANTS).set_index("n")

def get_constant(n, constant_name):
    """Retrieve unbiased constants for a given subgroup size."""
    if n in CONSTANTS_DF.index:
        return CONSTANTS_DF.loc[n, constant_name]
    else:
        # For n not in table, use interpolation or return None (usually n <= 25 covers most cases)
        return None

def calculate_capability(data, usl, lsl, subgroup_size=1):
    """
    Calculate Cp, Cpk (Short-term) and Pp, Ppk (Long-term).
    Short-term sigma is estimated based on within-group variation or moving range if n=1.
    """
    mu = np.mean(data)
    sigma_overall = np.std(data, ddof=1)
    
    # Estimate Sigma Within (Short-term)
    if subgroup_size > 1:
        # Using S-bar/c4 for estimate
        reshaped_data = data[:(len(data)//subgroup_size)*subgroup_size].reshape(-1, subgroup_size)
        s_bar = np.mean(np.std(reshaped_data, axis=1, ddof=1))
        c4 = get_constant(subgroup_size, "c4")
        sigma_within = s_bar / c4 if c4 else sigma_overall
    else:
        # For Individual measurements (n=1), use Average Moving Range / d2 (d2 for n=2)
        mr = np.abs(np.diff(data))
        mr_bar = np.mean(mr)
        d2 = get_constant(2, "d2")
        sigma_within = mr_bar / d2
    
    # Indices
    def calc_indices(s):
        cp = (usl - lsl) / (6 * s) if usl is not None and lsl is not None else np.nan
        cpu = (usl - mu) / (3 * s) if usl is not None else np.nan
        cpl = (mu - lsl) / (3 * s) if lsl is not None else np.nan
        cpk = np.nanmin([cpu, cpl]) if not (np.isnan(cpu) and np.isnan(cpl)) else np.nan
        return cp, cpk, cpu, cpl

    cp, cpk, cpu, cpl = calc_indices(sigma_within)
    pp, ppk, ppu, ppl = calc_indices(sigma_overall)
    
    return {
        "Mean": mu,
        "Sigma(Within)": sigma_within,
        "Sigma(Overall)": sigma_overall,
        "Cp": cp,
        "Cpk": cpk,
        "Pp": pp,
        "Ppk": ppk,
        "CPU": cpu,
        "CPL": cpl,
        "PPU": ppu,
        "PPL": ppl
    }

def normality_test(data):
    """Shapiro-Wilk test for normality."""
    if len(data) > 5000:
        # Shapiro-Wilk is limited to 5000 samples in some versions, or use Anderson-Darling
        stat, p = stats.anderson(data)
        return "Anderson-Darling", p # Simplified
    stat, p = stats.shapiro(data)
    return stat, p

def detect_nelson_rules(data, ucl, lcl, cl):
    """Detect basic out-of-control conditions (Nelson Rules 1-4)."""
    violations = []
    
    # Rule 1: One point beyond 3 sigma (UCL/LCL)
    for i, val in enumerate(data):
        if val > ucl or val < lcl:
            violations.append((i, "Rule 1: Beyond Limits"))
            
    # Rule 2: 9 points in a row on the same side of the center line
    side = np.sign(data - cl)
    for i in range(8, len(data)):
        if all(side[i-8:i+1] == side[i]):
            violations.append((i, "Rule 2: 9 points on one side"))
            
    # Rule 3: 6 points in a row steadily increasing or decreasing
    diffs = np.sign(np.diff(data))
    for i in range(5, len(diffs)):
        if all(diffs[i-5:i+1] == diffs[i-5]) and diffs[i-5] != 0:
            violations.append((i+1, "Rule 3: 6 points trending"))
            
    # Rule 4: 14 points in a row alternating up and down
    alt = np.diff(data) > 0
    for i in range(13, len(alt)):
        is_alt = True
        for j in range(i-12, i+1):
            if alt[j] == alt[j-1]:
                is_alt = False
                break
        if is_alt:
            violations.append((i+1, "Rule 4: 14 points alternating"))
            
    return violations

import pandas as pd
import numpy as np

def generate_sample():
    # Stable process
    np.random.seed(42)
    stable_data = np.random.normal(loc=10.0, scale=0.5, size=150)
    
    # Process with a shift (Out of control example)
    shift_data = np.random.normal(loc=10.0, scale=0.5, size=80)
    shift_data = np.append(shift_data, np.random.normal(loc=11.5, scale=0.5, size=20))
    
    # Process with high variance (Low Capability)
    incapable_data = np.random.normal(loc=10.0, scale=1.5, size=100)
    
    df_stable = pd.DataFrame({"Measurement": stable_data})
    df_shift = pd.DataFrame({"Measurement": shift_data})
    df_incapable = pd.DataFrame({"Measurement": incapable_data})
    
    df_stable.to_csv("sample_stable.csv", index=False)
    df_shift.to_csv("sample_out_of_control.csv", index=False)
    df_incapable.to_csv("sample_low_capability.csv", index=False)
    
    print("Sample CSV files generated: sample_stable.csv, sample_out_of_control.csv, sample_low_capability.csv")

if __name__ == "__main__":
    generate_sample()

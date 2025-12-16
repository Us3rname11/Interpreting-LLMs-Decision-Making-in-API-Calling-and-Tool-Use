"""
Calculates vectorized attribution metrics (Share, Density, Normalized Density) 
for 'Planning' and 'Action' phases based on pre-computed mass/length columns.

Metrics:
  1. Attribution Share (AS)   = Span_Mass / Total_Sequence_Mass
  2. Attribution Density (AD) = (Span_Mass / Span_Len) / Global_Mean_Intensity
  3. Normalized AD (NAD)      = AD / Sum(All_ADs_in_Sequence)  [Sums to 1.0]

Input:  ...results_dataframe.parquett ('sum_plan_attr', 'sum_action_attr', 'span_len')
Output: ..._all_metrics_computed.parquet (Adds 6 new columns: nad_*, attribution_share_*, attribution_density_*)
"""

import pandas as pd
import numpy as np
import os
import sys

# --- CONFIGURATION ---
DF_FILEPATH = "../data/results_dataframe.parquet"

def normalized_attribution_density(span_masses, span_lens, sequence_masses, sequence_lens):
    """
    Computes NAD: Normalizes the Attribution Density so the sum of spans = 1.0.
    """
    # 1. Setup inputs as float arrays
    span_masses = np.array(span_masses, dtype=float)
    span_lens = np.array(span_lens, dtype=float)
    sequence_masses = np.array(sequence_masses, dtype=float)
    sequence_lens = np.array(sequence_lens, dtype=float)
    
    # 2. Global Mean (Total Mass / Total Length)
    seq_total_mass = sequence_masses.sum()
    seq_total_len = sequence_lens.sum()
    
    if seq_total_len == 0 or seq_total_mass == 0:
        return np.full(span_masses.shape, np.nan)

    global_mean = seq_total_mass / seq_total_len
    
    # 3. Raw Attribution Density (Span Mean / Global Mean)
    with np.errstate(divide='ignore', invalid='ignore'):
        span_means = span_masses / span_lens
        raw_ads = span_means / global_mean
        
    # Handle NaNs (e.g., if a span has length 0)
    raw_ads = np.nan_to_num(raw_ads, nan=0.0)

    # 4. Normalize (Divide by sum of all ADs)
    total_ad_sum = raw_ads.sum()
    
    if total_ad_sum == 0:
        return np.zeros_like(raw_ads)
        
    return raw_ads / total_ad_sum

# --- MAIN EXECUTION ---
def compute_all_attribution_metrics():
    # 1. Load DataFrame
    print(f"Loading dataframe from {DF_FILEPATH}...")
    if not os.path.exists(DF_FILEPATH):
        print("Error: File not found.")
        return

    df = pd.read_parquet(DF_FILEPATH)
    print(f"Loaded {len(df)} rows.")

    # 2. Preparation
    # Ensure columns are float and handle NaNs (filling Mass NaNs with 0)
    df['sum_plan_attr'] = df['sum_plan_attr'].fillna(0.0).astype(float)
    df['sum_action_attr'] = df['sum_action_attr'].fillna(0.0).astype(float)
    df['span_len'] = df['span_len'].astype(float)

    # Initialize results storage
    results = {
        'nad_plan': pd.Series(index=df.index, dtype=float),
        'nad_action': pd.Series(index=df.index, dtype=float),
        
        'attribution_share_plan': pd.Series(index=df.index, dtype=float),
        'attribution_share_action': pd.Series(index=df.index, dtype=float),
        
        'attribution_density_plan': pd.Series(index=df.index, dtype=float),
        'attribution_density_action': pd.Series(index=df.index, dtype=float),
    }

    # 3. Grouping
    grouped = df.groupby('task_model_id')
    total_tasks = len(grouped)
    print(f"Found {total_tasks} unique tasks. Starting processing loop...")

    # 4. Processing Loop
    for i, (task_model_id, group) in enumerate(grouped):
        
        if i % 500 == 0:
            print(f"Processing task {i}/{total_tasks}...")

        span_lens = group['span_len'].values
        idx = group.index

        # --- Helper to calculate AS and AD inside the loop ---
        def calculate_phase_metrics(masses, lengths):
            # A. Normalized Attribution Density (NAD)
            nad = normalized_attribution_density(masses, lengths, masses, lengths)
            
            # B. Attribution Share (AS) = Mass / Total Mass
            total_mass = masses.sum()
            if total_mass == 0:
                as_vals = np.full(masses.shape, np.nan)
            else:
                as_vals = masses / total_mass
            
            # C. Attribution Density (AD) = (Mass/Len) / (TotalMass/TotalLen)
            total_len = lengths.sum()
            if total_len == 0 or total_mass == 0:
                ad_vals = np.full(masses.shape, np.nan)
            else:
                global_mean = total_mass / total_len
                with np.errstate(divide='ignore', invalid='ignore'):
                    span_means = masses / lengths
                    ad_vals = span_means / global_mean
                    # Fix artifacts where length was 0
                    ad_vals = np.nan_to_num(ad_vals, nan=0.0)

            return nad, as_vals, ad_vals

        # --- Calculate Plan Metrics ---
        plan_masses = group['sum_plan_attr'].values
        p_nad, p_as, p_ad = calculate_phase_metrics(plan_masses, span_lens)
        
        results['nad_plan'].loc[idx] = p_nad
        results['attribution_share_plan'].loc[idx] = p_as
        results['attribution_density_plan'].loc[idx] = p_ad

        # --- Calculate Action Metrics ---
        action_masses = group['sum_action_attr'].values
        a_nad, a_as, a_ad = calculate_phase_metrics(action_masses, span_lens)
        
        results['nad_action'].loc[idx] = a_nad
        results['attribution_share_action'].loc[idx] = a_as
        results['attribution_density_action'].loc[idx] = a_ad

    # 5. Save Results
    print("Processing complete. Assigning new columns...")
    for col, series in results.items():
        df[col] = series

    output_path = DF_FILEPATH.replace(".parquet", "_all_metrics_computed.parquet")
    print(f"Saving enriched DataFrame to {output_path}")
    df.to_parquet(output_path)
    
    # 6. Verification
    print("\n--- Summary of Computed Means ---")
    for col in results.keys():
        print(f" - {col}: {df[col].mean():.4f}")

    print(f"\nSanity Check (NAD Plan Sum per task, Target ~1.0): {df.groupby('task_model_id')['nad_plan'].sum().mean():.4f}")

if __name__ == "__main__":
    compute_all_attribution_metrics()
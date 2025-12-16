"""
Attribution AUPRC Computer
==========================

Purpose:
    Calculates the Area Under the Precision-Recall Curve (AUPRC) to quantify
    model focus during the planning phase. It measures how effectively the model
    attributed to the "Correct Tool" definition versus "Decoy Tools".

Methodology:
    1. Filter: Only processes tasks where the model successfully used the tool ('tool_use_correct' == True).
    2. Ground Truth (y_true): 
       - 1 (Positive): Token indices belonging to the 'correct_function' span.
       - 0 (Negative): Token indices belonging to 'decoy_functions' spans.
    3. Prediction (y_score): 
       - Sum of attention weights from the 'planning_step' tokens pointing to the target tokens.
    4. Calculation: Computes AUPRC based on the alignment between attention mass and ground truth.

Input:
    - DataFrame: ...results_dataframe.parquet
    - Attribution Files: .dill files containing raw attention matrices.

Output:
    - File: ..._auprc_computed.parquet
    - New Column: 'auprc' (Float score 0.0-1.0 for successful tasks, None otherwise).
"""


import pandas as pd
import numpy as np
import dill
import os
from sklearn.metrics import average_precision_score
from typing import Dict, List, Optional

# --- CONFIGURATION ---
DF_FILEPATH = "../data/results_dataframe.parquet"
ATTRIBUTION_DIR = "../trajectory_and_attribution_files/" 

def get_task_slices_and_mask(group_df: pd.DataFrame) -> Dict:
    """
    Parses a dataframe group to calculate slicing indices and ground truth mask.
    Supporting Evidence: Correct Tool definition tokens
    Opposing Evidence: Decoy Tool definitions tokens
    Output tokens: PLanning step
    """
    plan_row = group_df[group_df['span_name'] == 'planning_step']
    
    if plan_row.empty:
        return {'valid': False, 'error': 'Missing planning_step'}
    
    plan_len_val = plan_row.iloc[0]['span_len']
    if pd.isna(plan_len_val):
        return {'valid': False, 'error': 'planning_step length is NaN'}

    col_limit = int(plan_len_val)
    
    correct_rows = group_df[group_df['span_name'] == 'correct_function']
    decoy_rows = group_df[group_df['span_name'] == 'decoy_functions']
    
    if correct_rows.empty and decoy_rows.empty:
        return {'valid': False, 'error': 'No correct or decoy functions found'}

    correct_indices_set = set()
    decoy_indices_set = set()

    for _, row in correct_rows.iterrows():
        if pd.isna(row['span_start']) or pd.isna(row['span_end']):
            continue
        indices = range(int(row['span_start']), int(row['span_end']))
        correct_indices_set.update(indices)

    for _, row in decoy_rows.iterrows():
        if pd.isna(row['span_start']) or pd.isna(row['span_end']):
            continue
        indices = range(int(row['span_start']), int(row['span_end']))
        decoy_indices_set.update(indices)

    decoy_indices_set = decoy_indices_set - correct_indices_set
    all_indices_set = correct_indices_set.union(decoy_indices_set)
    
    if not all_indices_set:
        return {'valid': False, 'error': 'Spans had zero length or were all NaN'}

    sorted_row_indices = sorted(list(all_indices_set))
    
    y_true = []
    for idx in sorted_row_indices:
        if idx in correct_indices_set:
            y_true.append(1)
        else:
            y_true.append(0)
            
    return {
        'valid': True,
        'row_indices': sorted_row_indices, 
        'col_limit': col_limit,            
        'y_true': np.array(y_true)
    }

def load_and_compute_auprc(row_indices: List[int], 
                           col_limit: int, 
                           y_true: np.ndarray, 
                           filename_df: str) -> Optional[float]:
    """
    Loads the attribution matrix, slices it, sums over dim 1, and computes AUPRC.
    (Logic unchanged)
    """
    target_filename = filename_df.replace('agent_out.dill', 'inseq_out.dill')
    full_path = os.path.join(ATTRIBUTION_DIR, target_filename)

    if not os.path.exists(full_path):
        return None

    try:
        with open(full_path, 'rb') as file:
            loaded_data = dill.load(file)
            attribution_mat = loaded_data.sequence_attributions[0].target_attributions.numpy()

        sliced_mat = attribution_mat[row_indices, :col_limit]
        y_scores = np.sum(sliced_mat, axis=1)

        if len(y_scores) != len(y_true):
            print(f"  [!] Shape mismatch: y_scores {len(y_scores)} vs y_true {len(y_true)}")
            return None
        
        if len(np.unique(y_true)) < 2:
            pass 

        score = average_precision_score(y_true, y_scores)
        return score

    except Exception as e:
        print(f"  [!] Error processing file {target_filename}: {e}")
        return None

def main():
    # 1. Load DataFrame
    print(f"Loading dataframe from {DF_FILEPATH}...")
    try:
        df = pd.read_parquet(DF_FILEPATH)
    except Exception as e:
        print(f"Critical Error loading DataFrame: {e}")
        return

    print(f"Loaded {len(df)} rows. Grouping by task_model_id...")
    
    task_scores = {}
    grouped = df.groupby('task_model_id')
    total_tasks = len(grouped)
    
    print(f"Found {total_tasks} unique tasks. Starting processing loop...")
    
    # --- Counters ---
    skipped_span_error = 0
    skipped_incorrect = 0 
    
    for i, (task_model_id, group) in enumerate(grouped):
        
        if i % 100 == 0:
            print(f"Processing task {i}/{total_tasks}...")

        # --- STEP A: FILTER BY CORRECTNESS ---
        is_task_correct = group.iloc[0]['tool_use_correct']
        
        if not is_task_correct:
            task_scores[task_model_id] = None
            skipped_incorrect += 1
            continue
        # -------------------------------------

        # B. Get Slice Info & Mask
        slice_info = get_task_slices_and_mask(group)
        
        if not slice_info['valid']:
            task_scores[task_model_id] = None
            skipped_span_error += 1
            continue

        # C. Get Filename
        filename_from_df = group.iloc[0]['filename']

        # D. Load File and Compute Metric
        auprc = load_and_compute_auprc(
            row_indices=slice_info['row_indices'],
            col_limit=slice_info['col_limit'],
            y_true=slice_info['y_true'],
            filename_df=filename_from_df
        )
        
        task_scores[task_model_id] = auprc

    # 2. Store Results
    print("Processing complete. Mapping results back to DataFrame...")
    df['auprc'] = df['task_model_id'].map(task_scores)

    # 3. Save Results
    output_path = DF_FILEPATH.replace(".parquet", "_auprc_computed.parquet")
    print(f"Saving enriched DataFrame to {output_path}")
    df.to_parquet(output_path)
    
    # 4. Final Summary
    valid_scores = df['auprc'].dropna()
    valid_count = len(valid_scores)
    
    # Failed files are those that passed the logic checks but returned None from load_and_compute_auprc
    # Calculation: Total - (Skipped Incorrect + Skipped Spans + Successes)
    failed_files_count = total_tasks - skipped_incorrect - skipped_span_error - valid_count
    
    print("\n--- Run Summary ---")
    print(f"Total Tasks:               {total_tasks}")
    print(f"Skipped (Model Incorrect): {skipped_incorrect}")
    print(f"Skipped (Invalid Spans):   {skipped_span_error}")
    print(f"Failed (File/Calc Error):  {failed_files_count}")
    print(f"Successful Calculations:   {valid_count}")
    
    if not valid_scores.empty:
        print(f"Mean AUPRC (Correct Only): {valid_scores.mean():.4f}")
    else:
        print("Mean AUPRC:                N/A")

if __name__ == "__main__":
    main()
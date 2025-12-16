import pandas as pd
import dill
import os
import re

def compute_tool_metrics():
    # 1. Configuration / Paths
    input_parquet_path = "../data/results_dataframe.parquet"
    output_dir = "../trajectory_and_attribution_files/"
    
    # Define where you want to save the result
    output_parquet_path = "../data/new_dataframe_with_evaluation.parquet"

    print(f"Loading dataframe from: {input_parquet_path}")
    
    # 2. Load Dataframe
    try:
        df = pd.read_parquet(input_parquet_path)
    except Exception as e:
        print(f"Error loading dataframe: {e}")
        return

    # 3. Optimize Iteration
    unique_tasks = df[['task_model_id', 'filename', 'task']].drop_duplicates(subset=['task_model_id'])
    
    print(f"Found {len(df)} total rows.")
    print(f"Processing {len(unique_tasks)} unique task_model_ids...")

    # Dictionary to store results: {task_model_id: {'tool_plan_correct': bool, 'tool_use_correct': bool}}
    results_map = {}

    # 4. Main Loop
    for index, row in unique_tasks.iterrows():
        t_id = row['task_model_id']
        filename = row['filename']
        correct_tool = row['task']

        # Initialize defaults (False) in case file is missing or structure is wrong
        plan_correct = False
        use_correct = False

        full_file_path = os.path.join(output_dir, filename)

        try:
            # Check if file exists before trying to open
            if os.path.exists(full_file_path):
                
                # Open .dill file
                with open(full_file_path, 'rb') as file:
                    oneseven_data = dill.load(file)
                
                # Extract trace
                trace = oneseven_data[1]['model_output_message']['raw']
                
                # Ensure trace is string
                if not isinstance(trace, str):
                    trace = str(trace)

                # 5. Regex and Logic
                # re.escape ensures characters like '(', ')', '+', etc.
                # in 'correct_tool' are treated as literal characters, not regex commands.
                escaped_tool = re.escape(correct_tool)
                
                # Find the last appearance of "<code>"
                split_marker = "<code>"
                last_code_index = trace.rfind(split_marker)

                if last_code_index != -1:
                    # found <code>, split content
                    text_before_step = trace[:last_code_index]
                    text_after_step = trace[last_code_index:] 
                else:
                    # If <code> is not found at all
                    text_before_step = trace
                    text_after_step = ""

                # Search in plan (Before)
                if re.search(escaped_tool, text_before_step):
                    plan_correct = True

                # Search in action (After)
                if re.search(escaped_tool, text_after_step):
                    use_correct = True
            
            else:
                print(f"Warning: File not found for ID {t_id}: {full_file_path}")

        except Exception as e:
            print(f"Error processing ID {t_id} (File: {filename}): {e}")
        
        # Store results
        results_map[t_id] = {
            'tool_plan_correct': plan_correct,
            'tool_use_correct': use_correct
        }

    # 6. Merge Results back to Original Dataframe
    print("Processing complete. Merging results...")

    # Convert results dictionary to DataFrame for easy merging
    results_df = pd.DataFrame.from_dict(results_map, orient='index')
    
    # Merge on index (task_model_id)
    # The original df has task_model_id as a column, so we map using that
    df['tool_plan_correct'] = df['task_model_id'].map(lambda x: results_map.get(x, {}).get('tool_plan_correct', False))
    df['tool_use_correct'] = df['task_model_id'].map(lambda x: results_map.get(x, {}).get('tool_use_correct', False))

    # 7. Save Dataframe
    print(f"Saving updated dataframe to: {output_parquet_path}")
    df.to_parquet(output_parquet_path)
    print("Done.")

if __name__ == "__main__":
    compute_tool_metrics()
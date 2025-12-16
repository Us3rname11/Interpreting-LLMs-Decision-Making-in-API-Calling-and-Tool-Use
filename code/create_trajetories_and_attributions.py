"""
Agent Execution & Attribution Pipeline
======================================

Purpose:
    Runs an end-to-end evaluation and interpretability loop for LLM agents.
    For each task in a specific category, it:
    1.  **Executes the Agent:** Runs a `smolagents` CodeAgent with a set of tools to solve the task.
    2.  **Captures Trace:** Extracts the full conversation history (System -> User -> Thought/Action).
    3.  **Computes Attribution:** Uses `Inseq` to calculate how much the model attended to the prompt/tools 
        while generating its Plan and Action.
    4.  **Aggregates & Saves:** Reduces the high-dimensional attribution matrix (e.g., averaging heads/layers) 
        and saves both the agent logs and the attribution data.

Key Features:
    - **Checkpointing:** Automatically skips tasks that already have both `_agent_out.dill` and `_inseq_out.dill`.
    - **Dynamic Tool Loading:** Loads category-specific tools + random distractor tools from `tool_list_combined.json`.
    - **Inseq Integration:** Supports 'attention' and 'integrated_gradients' attribution methods.

Input:
    - Tasks: data/task_tool_dataset/dataset_combined.json
    - Tools: data/task_tool_dataset/tool_list_combined.json

Output:
    - Directory: outputs/{category}/
    - Files per task:
        1. `{id}_agent_out.dill`: Raw agent memory logs.
        2. `{id}_inseq_out.dill`: Aggregated attribution object (Target tokens x Source tokens).
"""

import os
import json
import random
import argparse
import dill
import torch
import glob
from typing import List, Tuple
from datetime import datetime
from smolagents import TransformersModel, CodeAgent, LogLevel
from smolagents.models import ChatMessage
from custom_tools import *
import inseq
from inseq.data.aggregator import AggregatorPipeline, SequenceAttributionAggregator
from transformers import AutoTokenizer


# ==============================================================================
# == 1. Function Definition
# ==============================================================================
def load_tasks_and_filter_solved(
    tasks_file: str,
    selected_category: str,
    output_dir: str,
    model_id_short: str,
    agent_suffix: str = "_agent_out.dill",
    inseq_suffix: str = "_inseq_out.dill",
) -> Tuple[List[dict], List[str], List[str]]:
    """
    Load tasks JSON, filter tasks by the category, and check OUTPUT_DIR for already-completed tasks.

    Args:
        tasks_file: Path to the JSON file containing tasks.
        selected_category: Category key whose tasks should be solved.
        output_dir: Path to dir where files are saved
        model_id_short: Model name
        agent_suffix: file ending to look for
        inseq_suffix: file ending to look for

    Returns:
        tasks_to_run: list of task records that still need to be processed
        solved_task_ids: list of task_id strings that are fully solved (both files found for a single unique_id)
        partial_task_ids: list of task_id strings for which partial output(s) were found (but not both)
    """
    # Load tasks JSON
    try:
        with open(tasks_file, "r") as f:
            all_task_records = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Tasks file not found at {tasks_file}")
        exit()

    # Filter by category
    tasks_in_category = [
        record for record in all_task_records if record.get("category") == selected_category
    ]

    if not tasks_in_category:
        print(f"ERROR: No tasks found for category '{selected_category}' in {tasks_file}")
        return [], [], []

    # Ensure output directory exists (it is created earlier in your script, but check anyway)
    if not os.path.isdir(output_dir):
        print(f"INFO: Output directory '{output_dir}' does not exist yet. Nothing is solved.")
        os.makedirs(output_dir, exist_ok=True)

    solved_task_ids = []
    partial_task_ids = []
    tasks_to_run = []

    # Gather list of files once
    out_files = set(os.listdir(output_dir))

    for i, record in enumerate(tasks_in_category):
        # Consistent fallback id
        raw_task_id = record.get("my_id", f"unknownID_{i}")
        task_id_str = str(raw_task_id)

        # Prefix used when writing unique_id-based filenames (timestamp appended after this)
        prefix = f"{selected_category}_ID{task_id_str}_{model_id_short}_"

        # Find files that begin with the prefix
        matching_files = [fn for fn in out_files if fn.startswith(prefix)]

        if not matching_files:
            # No outputs at all => schedule it
            tasks_to_run.append(record)
            continue

        # Build set of unique bases (unique_id without the final suffix)
        bases = set()
        for fn in matching_files:
            if fn.endswith(agent_suffix):
                bases.add(fn[: -len(agent_suffix)])
            elif fn.endswith(inseq_suffix):
                bases.add(fn[: -len(inseq_suffix)])
            else:
                # file with unexpected suffix, still include its filename as a base candidate (strip extension)
                base_no_ext = os.path.splitext(fn)[0]
                bases.add(base_no_ext)

        # For each base check whether both agent and inseq outputs exist for the same base
        found_full_pair = False
        for base in bases:
            agent_file = base + agent_suffix
            inseq_file = base + inseq_suffix
            if agent_file in out_files and inseq_file in out_files:
                found_full_pair = True
                break

        if found_full_pair:
            # Fully solved (at least one matching unique_id had both files)
            solved_task_ids.append(task_id_str)
            # Do NOT add to tasks_to_run
        else:
            # Partial outputs found (some files but no base had both endings). Warn and schedule the task.
            partial_task_ids.append(task_id_str)
            print(f"  -> WARNING: Partial outputs found for task ID {task_id_str}. No matching pair of '{agent_suffix}' + '{inseq_suffix}' for the same unique_id. This task will be re-run.")
            tasks_to_run.append(record)

    # Summary prints
    if solved_task_ids:
        print(f"\nAlready fully solved tasks ({len(solved_task_ids)}): {solved_task_ids}")
    else:
        print("\nNo fully solved tasks found.")

    if partial_task_ids:
        print(f"Tasks with partial outputs (will be re-run) ({len(partial_task_ids)}): {partial_task_ids}")

    print(f"Tasks to run (count={len(tasks_to_run)}).")

    return tasks_to_run#, solved_task_ids, partial_task_ids

def load_and_select_tools(tool_list_file: str, selected_category: str, num_random_tools: int = 0, seed: int = 42):
    """
    Load the tool-list JSON and prepare a list of callable tool functions for the
    given category. This is the refactored code originally present inline in
    the main execution block under the comment "# --- Load and Select Tools Dynamically ---".

    Args:
        tool_list_file: Path to the JSON file containing tools grouped by category.
        selected_category: Category key whose tools should be selected.
        num_random_tools: Number of random tools to sample and add from other categories (default=0).

    Returns:
        tool_functions: list of callable tool objects found in globals().
        final_tool_names: list of tool name strings that were considered.
        all_tools_by_category: parsed JSON mapping of categories -> tool name lists.
    """

    # Load tool list JSON
    try:
        with open(tool_list_file, 'r') as f:
            all_tools_by_category = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Tool list file not found at {tool_list_file}")
        exit()

    # Get the tools for the selected category
    tools_for_category = all_tools_by_category.get(selected_category)
    if not tools_for_category:
        print(f"ERROR: Category '{selected_category}' not found in {tool_list_file}")
        exit()
    print(f"Found {len(tools_for_category)} tools for category '{selected_category}'.")

    # Flatten all other tools
    other_tools = [
        tool_name
        for category, tool_list in all_tools_by_category.items()
        for tool_name in tool_list
        if category != selected_category
    ]

    # Set random seed if not passed
    if seed is not None:
        random.seed(seed)

    # Sample random tools from other categories (if requested)
    if len(other_tools) > num_random_tools:
        random_sample = random.sample(other_tools, num_random_tools)
        print(f"Adding a random sample of {len(random_sample)} tools from other categories (seed={seed}).")
    else:
        random_sample = other_tools  # Take all if fewer than requested
        print(f"Adding all {len(random_sample)} available tools from other categories (fewer than {num_random_tools}).")

    # Combine and deduplicate
    final_tool_names = list(set(tools_for_category + random_sample))

    # Map tool names (strings) to actual callable objects available in globals()
    tool_functions = []
    for tool_name in final_tool_names:
        tool_func = globals().get(tool_name)
        if tool_func and callable(tool_func):
            tool_functions.append(tool_func)
        else:
            print(f"  -> WARNING: Tool function '{tool_name}' not found and will be skipped.")

    return tool_functions, final_tool_names, all_tools_by_category

def run_my_agent(task_prompt: str, tools_to_use: list, model: TransformersModel) -> str:
    """
    Runs the LLM agent.
    
    Args:
        task_prompt: The input string (=task prompt) for the agent.
        
    Returns:
        The full log of the agent run as a list object.
    """
    my_tools = tools_to_use

    agent = CodeAgent(
        model=model, 
        #add_base_tools=True, 
        tools=my_tools,
        verbosity_level=LogLevel.INFO,
        max_steps=1
    )

    agent.run(task_prompt)

    agent_memory = agent.memory.get_full_steps()

    torch.cuda.empty_cache()

    return agent_memory


def prepare_for_inseq(memory_steps, tokenizer):
    """
    Extracts and formats the system + user prompts (formatted as the model sees them during inference)
    as well as the generated output from step 1 of smolagents memory.
    Preparation for inseq attribution.

    Args:
        memory_steps: list object as created by agent.memory.get_full_steps()
        tokenizer: A Transformers AutoTokenizer object
    
    Returns:
        input_text and output_text strings to be used by inseq
    """
    step1 = memory_steps[1]  # first step with actual model I/O
    messages: list[ChatMessage] = step1["model_input_messages"]
    assistant_msg: dict = step1["model_output_message"]

    # Flatten the content lists into strings
    def extract_text(msg: ChatMessage) -> str:
        if isinstance(msg.content, list):
            return "".join(
                part["text"] for part in msg.content if part["type"] == "text"
            )
        elif isinstance(msg.content, str):
            return msg.content
        return ""

    system_text = extract_text(next(m for m in messages if m.role.name == "SYSTEM"))
    user_text   = extract_text(next(m for m in messages if m.role.name == "USER"))

    # Format input exactly like Qwen sees it
    input_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        tokenize=False,
        add_generation_prompt=True,  # ensures <|assistant|> token is included
    )

    # Assistant output (already a string in smolagents memory)
    output_text = assistant_msg["content"]
    
    return input_text, output_text


def run_inseq_analysis(prompt: str, generated_text: str, inseq_model: inseq.AttributionModel) -> object:
    """
    Runs the Inseq attribution analysis on an agent run.
    
    Args:
        prompt: The original input prompt (system prompt + task prompt).
        generated_text: The text generated by the agent.
        
    Returns:
        The complex attribution output object from Inseq.
    """
    print("  -> Running Inseq analysis...")

    '''
    inseq_model = inseq.load_model(
        model_id,
        "attention",
        #"attention",
        #attn_implementation="eager"
    )#"saliency")#"integrated_gradients")
    '''

    attribution_result = inseq_model.attribute(
        prompt,
        generated_texts=prompt + generated_text,
        attribution_args={"n_steps": 10},
        internal_batch_size=1,
    )
    
    return attribution_result

def aggregate_inseq_output(attribution_result: object, method: str) -> object:
    """
    Squash an Inseq attribution result into 2D target_attributions [T, S], in-place.
    
    Args:
        attribution_result: The FeatureAttributionOutput from Inseq.
        method: Either "attention" or "integrated_gradients".
    
    Returns:
        The same attribution_result object, modified in-place so that
        .target_attributions are now of shape [nb_target_tokens, nb_source_tokens].
    """
    print("  -> Aggregating Inseq's output...")

    print(f"Attribution results shape before aggregation: {attribution_result.sequence_attributions[0].target_attributions.shape}")
    for seq_attr in attribution_result.sequence_attributions:
        if method == "attention":
            # Expect shape [T, S, num_layers, num_heads] -> average over the layers and attention heads
            seq_attr.target_attributions = seq_attr.target_attributions.mean(dim=(-1, -2))

        elif method == "integrated_gradients":
            # Expect shape [T, S, num_hidden_feature_dim] -> aggregate over the hidden feature dimension
            agg = AggregatorPipeline([SequenceAttributionAggregator()])
            aggregated = seq_attr.aggregate(aggregator=agg)
            seq_attr.target_attributions = aggregated.target_attributions

        else:
            raise ValueError(f"Unsupported method '{method}', choose 'attention' or 'integrated_gradients'.")
        
    print(f"Attribution results shape after aggregation: {attribution_result.sequence_attributions[0].target_attributions.shape}")

    return attribution_result


# ==============================================================================
# == 2. Execution Loop
# ==============================================================================

if __name__ == "__main__":
    # --- Setup Argument Parser ---
    parser = argparse.ArgumentParser(description="Run agent and Inseq analysis for a specific category of tasks.")
    parser.add_argument("--category", type=str, required=True, help="The category name from the dataset (e.g., 'entertainment and media').")
    parser.add_argument("--model_name", type=str, required=True, help="e.g. Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B")
    parser.add_argument("--attribution", type=str, required=True, help="Which attribution method is used by Inseq ('integrated_gradients', 'attention', 'saliency').")
    parser.add_argument("--tools", type=int, default=10, required=False, help="Optional: number of random tools to sample from other categories (default=10).")
    parser.add_argument("--start_id", type=str,required=False, help="Optional: Task ID to start execution with. Must exist in the selected category.")
    parser.add_argument("--seed", type=int, default=42, required=False, help="Optional: random seed for tool sampling (default=42).")
    args = parser.parse_args()

    # --- Select Model ID ---
    model_id_short = args.model_name
    model_id = "Qwen/" + model_id_short
    # --- Setup Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    selected_category = args.category
    
    # --- Setup Paths and Directories ---
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TASKS_FILE = os.path.join(parent_dir, "data", "task_tool_dataset", "dataset_combined.json")
    TOOL_LIST_FILE = os.path.join(parent_dir, "data", "task_tool_dataset", "tool_list_combined.json")
    OUTPUT_DIR = os.path.join(parent_dir, "outputs", selected_category)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Starting process for category: '{selected_category}'")
    
    # --- Load and Select Tools Dynamically ---
    num_random_tools = args.tools
    tool_functions, final_tool_names, all_tools_by_category = load_and_select_tools(TOOL_LIST_FILE, selected_category, num_random_tools, args.seed)
    
    # --- Load Tasks and Filter Solved Ones ---
    tasks_to_run = load_tasks_and_filter_solved( #tasks_to_run, solved_task_ids, partial_task_ids = load_tasks_and_filter_solved(
        TASKS_FILE,
        selected_category,
        OUTPUT_DIR,
        model_id_short,
        agent_suffix="_agent_out.dill",
        inseq_suffix="_inseq_out.dill",
    )

    if not tasks_to_run:
        print("No tasks to run. Exiting.")
        exit() 

    # --- Handle --start_id ---
    if args.start_id:
        start_id_str = str(args.start_id)
        available_ids = [str(t.get("my_id", f"unknownID_{i}")) for i, t in enumerate(tasks_to_run)]

        if start_id_str not in available_ids:
            print(f"ERROR: start_id '{start_id_str}' not found in pending tasks for category '{selected_category}'.")
            print(f"Available IDs in this category (not solved yet): {available_ids}")
            exit()

        # Slice tasks_to_run to start at the requested task
        start_index = next(
            i for i, t in enumerate(tasks_to_run)
            if str(t.get("my_id", f"unknownID_{i}")) == start_id_str
        )
        tasks_to_run = tasks_to_run[start_index:]
        print(f"Execution will start from task ID {start_id_str} (index {start_index}).")

    print(f"\n--- Loading model {model_id} into memory... ---")
    # Load the model for the agent
    agent_model = TransformersModel(
        model_id=model_id,
        device_map="cpu", #"auto"
        torch_dtype="auto"
    )
    
    inseq_model = inseq.load_model(
        model=agent_model.model,
        attribution_method=args.attribution,
        tokenizer=agent_model.tokenizer
    )
    
    # --- Main Processing Loop ---
    for i, task_record in enumerate(tasks_to_run):
        
        # 1. Extract the id and instruction from the record
        task_id = task_record.get("my_id", f"unknownID_{i}")#"id" # Fallback in case id is missing
        task_prompt = task_record.get("instruction", "")

        if not task_prompt:
            print(f"  -> WARNING: Skipping record with ID {task_id} due to missing 'instruction'.")
            continue

        print(f"\n--- Processing Task {i+1}/{len(tasks_to_run)} (ID: {task_id}) ---")
        
        timestamp = datetime.now().strftime("%d-%m_%H-%M-%S")
        unique_id = f"{selected_category}_ID{task_id}_{model_id_short}_{timestamp}"
        
        # 2. Run the agent
        agent_output_log = run_my_agent(task_prompt, tool_functions, agent_model)
        
        # 3. Save the agent's full log output
        agent_output_filename = os.path.join(OUTPUT_DIR, f"{unique_id}_agent_out.dill")
        with open(agent_output_filename, "wb") as f:
            dill.dump(agent_output_log, f)
        print(f"  -> Agent log saved to: {agent_output_filename}")
        
        # 4. Preprocess the models memory to be used as strings by inseq
        agent_input_string, agent_output_string = prepare_for_inseq(agent_output_log, tokenizer)
        
        # 5.1. Run the Inseq analysis
        inseq_output_object = run_inseq_analysis(agent_input_string, agent_output_string, inseq_model)

        # 5.2. Aggregate inseq's output by collapsing non-token-layers
        aggregated_inseq_output = aggregate_inseq_output(inseq_output_object, method=args.attribution)
        
        # 5.3 Save the aggregated Inseq object using dill
        inseq_output_filename = os.path.join(OUTPUT_DIR, f"{unique_id}_inseq_out.dill")
        with open(inseq_output_filename, "wb") as f:
            dill.dump(aggregated_inseq_output, f)
        print(f"  -> Inseq object saved to: {inseq_output_filename}")
        
    print("\n All tasks for the category completed successfully!")
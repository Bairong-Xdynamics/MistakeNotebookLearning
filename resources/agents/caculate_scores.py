import pandas as pd
import json
import argparse
import re
import sys
from collections import defaultdict

def parse_choice(text):
    """
    Parses the choice (e.g., 'A', 'B') from the answer text.
    Handles formats like:
    - "Answer: A"
    - "Choice: A"
    - "A."
    - "A"
    """
    if not isinstance(text, str):
        return None
    
    # Normalize
    text = text.strip()
    
    # Pattern 1: "Answer: X" (case insensitive)
    match = re.search(r'Answer:\s*([A-Z])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
        
    # Pattern 2: "Choice: X"
    match = re.search(r'Choice:\s*([A-Z])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
        
    # Pattern 3: Just a single letter "X" or "X." at the start
    # Be careful not to match sentences.
    # If the text starts with a letter followed by a dot or space or end of string.
    match = re.match(r'^([A-Z])(\.|$|\s)', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None

def load_data(file_paths):
    """
    Loads data from a list of file paths.
    """
    data = []
    if not file_paths:
        return data
    
    # If a single string is passed, wrap it in a list
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            print(f"Warning: Failed to decode line in {path}")
        except FileNotFoundError:
            print(f"Error: File not found {path}")
            sys.exit(1)
    return data

def evaluate_model(model_items, std_by_task):
    """
    Evaluates a single model dataset against the standard tasks.
    """
    # Map question -> model_item
    model_by_question = {}
    for item in model_items:
        q = item.get('question')
        if q:
            q_key = q.strip()
            model_by_question[q_key] = item
            
    select_correct_count = 0
    select_total_count = 0
    
    task_success_count = 0
    total_tasks = 0

    # New Metric: Step Acc based on explicit identifiers in model output
    step_acc_correct_count = 0
    step_acc_total_count = 0
    
    # Iterate over tasks defined in standard data
    for task_id, task_steps in std_by_task.items():
        is_task_success = True
        task_has_steps = False
        for std_step in task_steps:
            select_total_count += 1
            step_acc_total_count += 1
            task_has_steps = True
            q_key = std_step.get('question').strip()
            # Find corresponding model output
            model_step = model_by_question.get(q_key)
            # Extract ground truth choice
            gt_text = std_step.get('answer')
            gt_choice = parse_choice(gt_text)
            if model_step:
                # Extract model choice
                model_text = model_step.get('response')
                model_choice = parse_choice(model_text)
                if gt_choice and model_choice:
                    if gt_choice == model_choice:
                         select_correct_count += 1
                if model_step.get('result') > 0:
                    step_acc_correct_count += 1     
            if model_step.get('result') == 0:
                is_task_success = False
        if task_has_steps:
            total_tasks += 1
            if is_task_success:
                task_success_count += 1
                
    select_success_rate = (select_correct_count / select_total_count * 100) if select_total_count > 0 else 0.0
    task_success_rate = (task_success_count / total_tasks * 100) if total_tasks > 0 else 0.0
    
    step_acc_rate = 0.0
    if step_acc_total_count > 0:
        step_acc_rate = (step_acc_correct_count / step_acc_total_count * 100)
    
    return {
        'select_success_rate': select_success_rate,
        'select_correct_count': select_correct_count,
        'select_total_count': select_total_count,
        'task_success_rate': task_success_rate,
        'task_success_count': task_success_count,
        'total_tasks': total_tasks,
        'step_acc_rate': step_acc_rate,
        'step_acc_correct_count': step_acc_correct_count,
        'step_acc_total_count': step_acc_total_count
    }

def main():
    parser = argparse.ArgumentParser(description='Calculate Mind2Web Scores')
    parser.add_argument('--standard_files', nargs='+', required=False, help='Path to standard answer files (jsonl)', default="./resources/agents/mind2web_topk20_100_eval.jsonl")
    parser.add_argument('--model_files', nargs='+', required=False, help='Path to model output files (jsonl)', default="./examples/results/eval_wrong_cases.jsonl")
    
    args = parser.parse_args()
    
    # Load Standard Data
    print("Loading standard files...")
    standard_items = load_data(args.standard_files)
    
    if not standard_items:
        print("No standard items loaded.")
        return

    # Map question -> standard_item and group by task
    std_by_task = defaultdict(list)
    
    for item in standard_items:
        task_id = item.get('task_id')
        if task_id:
            std_by_task[task_id].append(item)
            
    print(f"Loaded {len(standard_items)} standard items.")
    print(f"Found {len(std_by_task)} unique tasks.")
    
    # Process each model file separately
    if not args.model_files:
        print("No model files provided.")
        return

    # Handle if args.model_files is a string (should be list due to nargs='+')
    model_files_list = args.model_files if isinstance(args.model_files, list) else [args.model_files]

    for model_file in model_files_list:
        print(f"\nEvaluating file: {model_file}")
        model_items = load_data([model_file])
        
        if not model_items:
            print(f"No items loaded from {model_file}")
            continue
            
        print(f"Loaded {len(model_items)} items from {model_file}")
        
        results = evaluate_model(model_items, std_by_task)
        
        print("-" * 30)
        print(f"File: {model_file}")
        print(f"Select Success (Element Match): {results['select_success_rate']:.2f}% ({results['select_correct_count']}/{results['select_total_count']})")
        print(f"Task Success (All Steps Correct): {results['task_success_rate']:.2f}% ({results['task_success_count']}/{results['total_tasks']})")
        if results['step_acc_total_count'] > 0:
            print(f"Step Acc (Direct from Model Identifier): {results['step_acc_rate']:.2f}% ({results['step_acc_correct_count']}/{results['step_acc_total_count']})")
        else:
            print("Step Acc (Direct from Model Identifier): N/A (No identifier found in model output)")
        print("-" * 30)

if __name__ == "__main__":
    main()

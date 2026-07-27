#!/usr/bin/env python3
"""
Use OpenAI API to fix problematic code and extract objective values.
Finds problematic code based on log files without "is correct:".
"""

import json
import sys
import glob
import os
import subprocess
import tempfile
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

def read_code_file(code_path):
    """Read the generated code file."""
    with open(code_path, 'r') as f:
        return f.read()

def fix_code_with_llm(code_content):
    """
    Use OpenAI API to fix the code.
    Returns the corrected code.
    """
    prompt = f"""Modify this optimization code to return only the objective value.
IMPORTANT: Keep all the code logic exactly the same (model setup, variables, constraints, objective, solve).
Only change the final return statement to return the objective value.

Do not change anything else. Only modify the return line.

Original code:
{code_content}

Return only the corrected Python code, no explanations."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        fixed_code = response.choices[0].message.content

        # Clean up markdown code blocks if present
        if fixed_code.startswith("```"):
            fixed_code = "\n".join(fixed_code.split("\n")[1:])
        if fixed_code.endswith("```"):
            fixed_code = "\n".join(fixed_code.split("\n")[:-1])

        return fixed_code.strip(), None
    except Exception as e:
        return None, f"LLM Error: {str(e)}"

def execute_code(code_content, code_idx):
    """
    Execute the fixed code in an isolated subprocess and extract the objective value.
    Returns (objective_value, error_message)
    """
    # Create a wrapper script that runs the code and prints the result
    wrapper_code = f'''
import sys
import json

{code_content}

if __name__ == "__main__":
    try:
        result = prob_solution()
        if isinstance(result, tuple):
            print(json.dumps({{"error": f"Expected single value, got tuple: {{result}}"}}))
        else:
            print(json.dumps({{"objective": result}}))
    except Exception as e:
        print(json.dumps({{"error": f"{{type(e).__name__}}: {{str(e)}}"}}))
'''

    try:
        # Run in isolated subprocess with timeout
        result = subprocess.run(
            [sys.executable, "-c", wrapper_code],
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            return None, f"Subprocess error: {stderr[:500]}"

        # Parse the JSON output
        output = result.stdout.strip()
        stderr = result.stderr.strip()

        if not output:
            if stderr:
                return None, f"No stdout, stderr: {stderr[:500]}"
            return None, "No output from subprocess"

        # Try to find JSON in the output (might have other prints before it)
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            # Try to find JSON object in the last line
            lines = output.strip().split('\n')
            for line in reversed(lines):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        data = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
            else:
                # No valid JSON found
                return None, f"No JSON in output. stdout: {output[:300]}, stderr: {stderr[:200]}"

        if "error" in data:
            return None, data["error"]
        return data.get("objective"), None

    except subprocess.TimeoutExpired:
        return None, "Execution timeout (60s)"
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)}"

def get_problematic_code_indices(log_dir):
    """
    Find all log files without "is correct:" and return their code indices.
    """
    pattern = os.path.join(log_dir, "*_test_log.txt")
    log_files = sorted(glob.glob(pattern))

    problematic_indices = []

    for filepath in log_files:
        filename = os.path.basename(filepath)

        with open(filepath, 'r') as f:
            content = f.read()
            if "is correct:" not in content:
                # Extract code index from filename (e.g., "42_test_log.txt" -> "42")
                try:
                    code_idx = filename.split('_')[0]
                    problematic_indices.append(code_idx)
                except:
                    pass

    return problematic_indices

def process_single_code(code_idx, base_dir, cleaned_codes_dir):
    """
    Process a single code file: fix, save, and extract objective.
    Returns (code_idx, objective, error)
    """
    code_file = base_dir / code_idx / "generated_code.py"

    if not code_file.exists():
        return code_idx, None, "File not found"

    try:
        # Read original code
        original_code = read_code_file(code_file)

        # Fix with LLM
        fixed_code, llm_error = fix_code_with_llm(original_code)

        if fixed_code is None:
            return code_idx, None, llm_error

        # Save fixed code to cleaned_codes directory
        cleaned_code_file = cleaned_codes_dir / code_idx / "generated_code.py"
        cleaned_code_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cleaned_code_file, 'w') as f:
            f.write(fixed_code)

        # Execute and extract objective
        objective, exec_error = execute_code(fixed_code, code_idx)

        if exec_error:
            return code_idx, None, exec_error
        else:
            return code_idx, objective, None

    except Exception as e:
        return code_idx, None, f"Processing error: {str(e)}"

def load_ground_truth(dataset_path):
    """
    Load ground truth objective values from the NL4OPT dataset.
    Returns dict mapping code_idx to objective value.
    """
    ground_truth = {}
    dataset_dir = Path(dataset_path)

    for idx_dir in sorted(dataset_dir.iterdir()):
        if idx_dir.is_dir():
            solution_file = idx_dir / "solution.json"
            if solution_file.exists():
                try:
                    with open(solution_file, 'r') as f:
                        data = json.load(f)
                        ground_truth[idx_dir.name] = data.get("objective")
                except Exception as e:
                    print(f"Warning: Could not load ground truth for {idx_dir.name}: {e}")

    return ground_truth

def main():
    task = "ComplexLP"
    model = "qwen/qwen3-30b-a3b-thinking-2507"
    log_dir = f"/hpc/group/fanglab/xx102/Chain-of-Experts/log/{task}_{model}"
    base_dir = Path(f"{log_dir}/codes")
    cleaned_codes_dir = Path(f"{log_dir}/cleaned_codes")
    results_file = Path(f"{log_dir}/fix_results.json")
    validation_file = Path(f"{log_dir}/validation_results.json")
    dataset_path = f"/hpc/group/fanglab/xx102/Chain-of-Experts/dataset/{task}"

    # Create cleaned_codes directory if it doesn't exist
    cleaned_codes_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth
    print("Loading ground truth from dataset...")
    ground_truth = load_ground_truth(dataset_path)
    print(f"Loaded ground truth for {len(ground_truth)} entries\n")

    # Find problematic code indices from log files
    problematic_indices = get_problematic_code_indices(log_dir)
    print(f"Found {len(problematic_indices)} problematic code files (without 'is correct:')...")

    fixed_count = 0
    successful = {}
    still_failed = {}

    # Use ThreadPoolExecutor - each thread spawns isolated subprocess for code execution
    num_workers = 16  # Adjust this based on your system capacity
    print(f"Processing with {num_workers} threads (each with isolated subprocess)...\n")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_single_code, code_idx, base_dir, cleaned_codes_dir): code_idx
            for code_idx in problematic_indices
        }

        # Process completed tasks
        completed = 0
        for future in as_completed(futures):
            completed += 1
            code_idx = futures[future]

            try:
                code_idx, objective, error = future.result()

                if error is None:
                    successful[code_idx] = objective
                    fixed_count += 1
                    print(f"  [{completed}/{len(problematic_indices)}] ✓ Code {code_idx}: Objective = {objective}")
                else:
                    still_failed[code_idx] = error
                    print(f"  [{completed}/{len(problematic_indices)}] ✗ Code {code_idx}: {error}")
            except Exception as e:
                still_failed[code_idx] = f"Process crashed: {type(e).__name__}: {str(e)}"
                print(f"  [{completed}/{len(problematic_indices)}] ✗ Code {code_idx}: Process crashed - {type(e).__name__}: {str(e)}")

    # Save results
    results_data = {
        "successful": successful,
        "failed": still_failed,
        "summary": {
            "total": len(successful) + len(still_failed),
            "successful": len(successful),
            "failed": len(still_failed)
        }
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    # Validate predictions against ground truth
    validation_data = {}
    correct_count = 0

    for code_idx, predicted_obj in successful.items():
        ground_truth_obj = ground_truth.get(code_idx)
        is_correct = False

        if ground_truth_obj is not None:
            # Check if objective values match (with small tolerance for floating point)
            try:
                is_correct = abs(float(predicted_obj) - float(ground_truth_obj)) < 0.1
                if is_correct:
                    correct_count += 1
            except (ValueError, TypeError) as e:
                # Record the terms that failed conversion and set is_correct to False
                print(f"Warning: Could not compare values for code {code_idx}: predicted={predicted_obj}, ground_truth={ground_truth_obj} - {e}")
                is_correct = False

        validation_data[code_idx] = {
            "predicted": predicted_obj,
            "ground_truth": ground_truth_obj,
            "correct": is_correct
        }

    # Save validation results
    with open(validation_file, 'w') as f:
        json.dump(validation_data, f, indent=2)

    print(f"\nDone!")
    print(f"  Fixed: {fixed_count}")
    print(f"  Still failed: {len(still_failed)}")
    print(f"  Total: {len(successful)}/{len(successful) + len(still_failed)}")
    print(f"  Correct predictions: {correct_count}/{len(successful)}")
    print(f"\nResults saved to: {results_file}")
    print(f"Validation results saved to: {validation_file}")
    print(f"Fixed codes saved to: {cleaned_codes_dir}")

if __name__ == "__main__":
    main()

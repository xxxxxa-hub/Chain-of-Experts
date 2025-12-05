#!/usr/bin/env python3
"""
Use OpenAI API to fix problematic code and extract objective values.
Finds problematic code based on log files without "is correct:".
"""

import json
import sys
import glob
import os
from pathlib import Path
import importlib.util
from openai import OpenAI

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
Only change the final return statement to:
  return m.objVal
(or m.ObjVal if that's what's used in the code)

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
    Execute the fixed code and extract the objective value.
    Returns (objective_value, error_message)
    """
    try:
        # Create a temporary module
        spec = importlib.util.spec_from_loader(f"fixed_code_{code_idx}", loader=None)
        module = importlib.util.module_from_spec(spec)

        # Execute the code
        exec(code_content, module.__dict__)

        # Call the function
        result = module.prob_solution()

        # Extract objective (first return value or single value)
        if isinstance(result, tuple):
            objective = result[0]
        else:
            objective = result

        return objective, None
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

def main():
    log_dir = "/hpc/group/fanglab/xx102/Chain-of-Experts/log/run_coe_NL4OPT_1764960639"
    base_dir = Path(f"{log_dir}/codes")
    cleaned_codes_dir = Path(f"{log_dir}/cleaned_codes")
    results_file = Path(f"{log_dir}/fix_results.json")

    # Create cleaned_codes directory if it doesn't exist
    cleaned_codes_dir.mkdir(parents=True, exist_ok=True)

    # Find problematic code indices from log files
    problematic_indices = get_problematic_code_indices(log_dir)
    breakpoint()
    print(f"Found {len(problematic_indices)} problematic code files (without 'is correct:')...")

    fixed_count = 0
    successful = {}
    still_failed = {}

    for i, code_idx in enumerate(problematic_indices):
        code_file = base_dir / code_idx / "generated_code.py"

        if not code_file.exists():
            still_failed[code_idx] = "File not found"
            continue


        print(f"  [{i+1}/{len(problematic_indices)}] Processing code {code_idx}...")

        try:
            # Read original code
            original_code = read_code_file(code_file)

            # Fix with LLM
            fixed_code, llm_error = fix_code_with_llm(original_code)

            if fixed_code is None:
                still_failed[code_idx] = llm_error
                continue

            # Save fixed code to cleaned_codes directory
            cleaned_code_file = cleaned_codes_dir / code_idx / "generated_code.py"
            cleaned_code_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cleaned_code_file, 'w') as f:
                f.write(fixed_code)

            # Execute and extract objective
            objective, exec_error = execute_code(fixed_code, code_idx)

            if exec_error:
                still_failed[code_idx] = exec_error
            else:
                # Success!
                successful[code_idx] = objective
                fixed_count += 1
                print(f"    ✓ Fixed! Objective: {objective}")

        except Exception as e:
            still_failed[code_idx] = f"Processing error: {str(e)}"

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

    print(f"\nDone!")
    print(f"  Fixed: {fixed_count}")
    print(f"  Still failed: {len(still_failed)}")
    print(f"  Total: {len(successful)}/{len(successful) + len(still_failed)}")
    print(f"\nResults saved to: {results_file}")
    print(f"Fixed codes saved to: {cleaned_codes_dir}")

if __name__ == "__main__":
    main()

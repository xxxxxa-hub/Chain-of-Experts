#!/usr/bin/env python3
"""
Script to categorize test log files by "is correct:" status
- is correct: True
- is correct: False
- is correct: MISSING
"""

import glob
import os

def print_problematic_logs(directory):
    """
    Categorize test log files by 'is correct:' status.
    Focuses on files without "is correct:" (problematic files).
    """

    # Find all test log files
    pattern = os.path.join(directory, "*_test_log.txt")
    log_files = sorted(glob.glob(pattern))

    if not log_files:
        print(f"No test log files found in {directory}")
        return

    files_with_correct_true = []
    files_with_correct_false = []
    files_with_correct_missing = []

    # Categorize files by is correct status
    for filepath in log_files:
        filename = os.path.basename(filepath)

        with open(filepath, 'r') as f:
            content = f.read()

        try:
            code_idx = filename.split('_')[0]
        except:
            continue

        if "is correct: True" in content:
            files_with_correct_true.append((code_idx, filepath))
        elif "is correct: False" in content:
            files_with_correct_false.append((code_idx, filepath))
        elif "is correct:" not in content:
            files_with_correct_missing.append((code_idx, filepath))

    # Print summary
    print("\n" + "=" * 80)
    print("IS CORRECT: STATUS SUMMARY")
    print("=" * 80)
    print(f"is correct: True:   {len(files_with_correct_true)}")
    print(f"is correct: False:  {len(files_with_correct_false)}")
    print(f"is correct: MISSING: {len(files_with_correct_missing)} (PROBLEMATIC)")
    print(f"Total files:        {len(log_files)}")
    print("=" * 80)

    # Print problematic files (missing is correct)
    print(f"\nDETAIL: {len(files_with_correct_missing)} Files with MISSING 'is correct:'")
    print("=" * 80)

    for code_idx, filepath in files_with_correct_missing[:20]:  # Print first 20
        filename = os.path.basename(filepath)

        print(f"\nFILE: {filename}")
        print("-" * 80)

        with open(filepath, 'r') as f:
            print(f.read())

    if len(files_with_correct_missing) > 20:
        print(f"\n... and {len(files_with_correct_missing) - 20} more files with missing 'is correct:'")

if __name__ == "__main__":
    task = "ComplexLP"
    model = "o4-mini"
    directory = f"/hpc/group/fanglab/xx102/Chain-of-Experts/log/{task}_{model}"
    print_problematic_logs(directory)

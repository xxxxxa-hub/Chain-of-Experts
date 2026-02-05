import argparse
import time
import os
import re
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from langchain.callbacks import get_openai_callback
from test_generated_code import test_generated_code, get_ground_truth
from utils import extract_code_from_string, read_problem
from result import Result
import baseline.standard as standard
import baseline.chain_of_thought as cot
import baseline.progressive_hint as php
from main import chain_of_experts


algorithms = {
    'standard': standard,
    'chain_of_thought': cot,
    'cot': cot,
    'progressive_hint': php,
    'php': php,
    # 'solo_performance_prompting': ssp,
    # 'ssp': ssp,
    # 'reflexion': reflexion,
}


def process_problem(args_dict):
    """Worker function to process a single problem.

    Args:
        args_dict: Dictionary containing problem data and arguments

    Returns:
        Dictionary with results (passed, error_type)
    """
    problem = args_dict['problem']
    dataset = args_dict['dataset']
    algorithm = args_dict['algorithm']
    path = args_dict['path']
    model = args_dict['model']
    max_collaborate_nums = args_dict['max_collaborate_nums']
    enable_reflection = args_dict['enable_reflection']
    max_trials = args_dict['max_trials']

    try:
        print(f"Processing: {problem}")
        problem_data = read_problem(dataset, problem)

        with get_openai_callback() as cb:
            if algorithm == 'chain_of_experts' or algorithm == 'coe':
                answer = chain_of_experts(
                    problem_data,
                    max_collaborate_nums,
                    model_name=model,
                    enable_reflection=enable_reflection,
                    max_trials=max_trials)
                time.sleep(10)
            else:
                algo_module = algorithms[algorithm]
                answer = algo_module.solve(problem_data, model_name=model)
            print('-' * 10 + 'Token usage' + '-' * 20)
            print(cb)
            print('-' * 25)

        # Write original answer to file
        with open(os.path.join(path, f'{problem}_original_answer.txt'), 'w', encoding='utf8') as f:
            f.write(answer)

        code = extract_code_from_string(answer)

        # Write code to problem-specific directory (thread-safe)
        problem_code_dir = os.path.join(path, 'codes', problem)
        Path(problem_code_dir).mkdir(parents=True, exist_ok=True)
        code_file_path = os.path.join(problem_code_dir, 'generated_code.py')
        with open(code_file_path, 'w', encoding='utf8') as f:
            f.write(code)

        # Also save to the standard location for logging
        with open(os.path.join(path, f'{problem}_generated_code.py'), 'w', encoding='utf8') as f:
            f.write(code)

        # Test the generated code with the specific code file
        ground_truth = get_ground_truth(dataset, problem)
        with open(os.path.join(path, f'{problem}_test_log.txt'), 'w', encoding='utf8') as f:
            result = test_generated_code(problem, ground_truth, f, generated_code_path=code_file_path)

        return {
            'problem': problem,
            'result': result,
            'success': True
        }

    except Exception as e:
        print(f"Error processing {problem}: {str(e)}")
        return {
            'problem': problem,
            'result': None,
            'success': False,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Generate and test code.')
    parser.add_argument('--dataset', type=str, help='Dataset name, "LPWP" or "ComplexOR"')
    parser.add_argument('--problem', type=str, help='Problem name')
    parser.add_argument('--algorithm', type=str, help='Algorithm name')
    parser.add_argument('--enable_reflection', action='store_true', help='Enable reflection option')
    parser.add_argument('--log_dir', type=str, default='log', help='The directory of log')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='Base large language model')
    parser.add_argument('--max_collaborate_nums', type=int, default=3, help='Number of max collaborations')
    parser.add_argument('--max_trials', type=int, default=3, help='Maximum number of forward-backward trials')
    args = parser.parse_args()
    args.algorithm = args.algorithm.lower()

    matched_problems = []
    for p in os.listdir(os.path.join('dataset', args.dataset)):
        if re.match(args.problem, p):
            matched_problems.append(p)
    total_num = len(matched_problems)
    if total_num == 0:
        print('No problem matched! Please check arguements.')
        exit(0)

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    log_dir_name = f'run_{args.algorithm}_{args.dataset}_{str(round(time.time()))}'
    path = os.path.join(args.log_dir, log_dir_name)
    print(f'Save log to {path}')
    Path(path).mkdir(parents=True, exist_ok=True)

    # Prepare arguments for each problem
    problem_args = []
    for problem in matched_problems:
        problem_args.append({
            'problem': problem,
            'dataset': args.dataset,
            'algorithm': args.algorithm,
            'path': path,
            'model': args.model,
            'max_collaborate_nums': args.max_collaborate_nums,
            'enable_reflection': args.enable_reflection,
            'max_trials': args.max_trials,
        })

    # Use thread pool (adjust num_processes based on your system)
    num_processes = min(50, len(matched_problems))
    correct_num = 0
    ce_num = 0
    re_num = 0
    current_num = 0

    print(f"Running with {num_processes} processes...")
    with Pool(processes=num_processes) as pool:
        with tqdm(total=len(matched_problems)) as pbar:
            for result_dict in pool.imap_unordered(process_problem, problem_args):

                if result_dict['success']:
                    result = result_dict['result']
                    if result == Result.ACCEPT:
                        correct_num += 1
                    elif result == Result.COMPILE_ERROR:
                        ce_num += 1
                    elif result == Result.RUNTIME_ERROR:
                        re_num += 1
                else:
                    re_num += 1  # Count failed problems as runtime errors

                current_num += 1
                pbar.update()
                pbar.set_description(
                    f'Accuracy: {correct_num / current_num * 100:.2f}% | '
                    f'Compile error: {ce_num / current_num * 100:.2f}% | '
                    f'Runtime error: {re_num / current_num * 100:.2f}%'
                )

    print(f'Passed: {correct_num}/{total_num}')
    print(f'Accuracy: {correct_num / total_num * 100:.2f}%')
    print(f'Compile error: {ce_num / total_num * 100:.2f}%')
    print(f'Runtime error: {re_num / total_num * 100:.2f}%')

if __name__ == '__main__':
    main()

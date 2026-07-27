import argparse
import json
import time
import os
import re
import traceback
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
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
}


def process_problem(args_dict):
    """Worker function to process a single problem.

    Returns:
        Dictionary with per-problem results including all four metrics.
    """
    problem = args_dict['problem']
    dataset = args_dict['dataset']
    algorithm = args_dict['algorithm']
    path = args_dict['path']
    model = args_dict['model']
    max_collaborate_nums = args_dict['max_collaborate_nums']
    enable_reflection = args_dict['enable_reflection']
    max_trials = args_dict['max_trials']

    start_time = time.time()

    try:
        print(f"Processing: {problem}")
        problem_data = read_problem(dataset, problem)

        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_cost = 0.0
        api_call_count = 0

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
            total_tokens = cb.total_tokens
            prompt_tokens = cb.prompt_tokens
            completion_tokens = cb.completion_tokens
            api_call_count = cb.successful_requests
            # OpenRouter pricing for gemini-2.5-flash: $0.30/M input, $2.50/M output
            # (langchain callback returns $0 for OpenRouter endpoints)
            PRICING = {
                "google/gemini-2.5-flash": (3e-7, 2.5e-6),
            }
            input_price, output_price = PRICING.get(model, (3e-7, 2.5e-6))
            total_cost = prompt_tokens * input_price + completion_tokens * output_price

        elapsed_time = time.time() - start_time

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
            'result': result.name,  # ACCEPT, WRONG_ANSWER, RUNTIME_ERROR, COMPILE_ERROR
            'ground_truth': ground_truth,
            'success': True,
            'elapsed_time': elapsed_time,
            'api_call_count': api_call_count,
            'total_tokens': total_tokens,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_cost': total_cost,
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"Error processing {problem}: {str(e)}")
        traceback.print_exc()
        return {
            'problem': problem,
            'result': 'RUNTIME_ERROR',
            'ground_truth': None,
            'success': False,
            'error': str(e),
            'elapsed_time': elapsed_time,
            'api_call_count': 0,
            'total_tokens': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_cost': 0.0,
        }


def main():
    parser = argparse.ArgumentParser(description='Generate and test code with four-metric recording.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., BWOR, ComplexLP)')
    parser.add_argument('--problem', type=str, default='.*', help='Problem name regex pattern')
    parser.add_argument('--algorithm', type=str, required=True, help='Algorithm name')
    parser.add_argument('--enable_reflection', action='store_true', help='Enable reflection option')
    parser.add_argument('--log_dir', type=str, default='log', help='The directory of log')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='Base large language model')
    parser.add_argument('--max_collaborate_nums', type=int, default=3, help='Number of max collaborations')
    parser.add_argument('--max_trials', type=int, default=3, help='Maximum number of forward-backward trials')
    parser.add_argument('--max_problems', type=int, default=10, help='Maximum number of problems to run (default: 10)')
    parser.add_argument('--num_processes', type=int, default=None, help='Number of parallel processes (default: min(50, num_problems))')
    parser.add_argument('--resume_dir', type=str, default=None, help='Resume from an existing log directory')
    parser.add_argument('--output', type=str, default=None, help='Output JSONL file path for per-problem results')
    args = parser.parse_args()
    args.algorithm = args.algorithm.lower()

    # Collect and sort matched problems
    matched_problems = []
    for p in os.listdir(os.path.join('dataset', args.dataset)):
        if re.match(args.problem, p):
            matched_problems.append(p)

    # Sort problems numerically (handle mixed naming)
    def sort_key(name):
        # Extract numeric part for sorting
        nums = re.findall(r'\d+', name)
        return int(nums[0]) if nums else name
    matched_problems.sort(key=sort_key)

    if len(matched_problems) == 0:
        print('No problem matched! Please check arguments.')
        exit(0)

    # Limit to first N problems
    if args.max_problems and len(matched_problems) > args.max_problems:
        print(f'Limiting to first {args.max_problems} of {len(matched_problems)} matched problems')
        matched_problems = matched_problems[:args.max_problems]

    total_num = len(matched_problems)
    print(f'Running {total_num} problems from dataset {args.dataset}')

    # Determine log directory
    if args.resume_dir:
        path = args.resume_dir
        if not os.path.isdir(path):
            print(f'Resume directory does not exist: {path}')
            exit(1)
        skipped = []
        remaining = []
        for p in matched_problems:
            if os.path.exists(os.path.join(path, f'{p}_test_log.txt')):
                skipped.append(p)
            else:
                remaining.append(p)
        print(f'Resuming from {path}')
        print(f'  Already completed: {len(skipped)} problems')
        print(f'  Remaining: {len(remaining)} problems')
        matched_problems = remaining
        if not matched_problems:
            print('All problems already completed. Nothing to do.')
            exit(0)
    else:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        log_dir_name = f'{args.dataset}_{args.model.replace("/", "_")}_re_run'
        path = os.path.join(args.log_dir, log_dir_name)
        Path(path).mkdir(parents=True, exist_ok=True)

    print(f'Save log to {path}')

    # Set up output JSONL path
    if args.output is None:
        output_path = os.path.join(path, f'results_{args.algorithm}_{args.dataset}_{args.model.replace("/", "_")}_re_run.jsonl')
    else:
        output_path = args.output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

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

    # Track all four metrics
    num_processes = args.num_processes or min(50, len(matched_problems))
    accept_num = 0
    wrong_answer_num = 0
    compile_error_num = 0
    runtime_error_num = 0
    current_num = 0
    total_elapsed = 0.0
    total_tokens_all = 0
    total_cost_all = 0.0

    # Truncate the JSONL file so previous runs are overwritten
    with open(output_path, 'w', encoding='utf-8') as out_f:
        pass

    print(f"Running with {num_processes} threads...")
    print(f"Results will be saved to: {output_path}")

    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        futures = {executor.submit(process_problem, pa): pa for pa in problem_args}
        with tqdm(total=len(matched_problems)) as pbar:
            for future in as_completed(futures):
                result_dict = future.result()
                result_name = result_dict['result']

                if result_name == 'ACCEPT':
                    accept_num += 1
                elif result_name == 'WRONG_ANSWER':
                    wrong_answer_num += 1
                elif result_name == 'COMPILE_ERROR':
                    compile_error_num += 1
                elif result_name == 'RUNTIME_ERROR':
                    runtime_error_num += 1

                current_num += 1
                total_elapsed += result_dict.get('elapsed_time', 0)
                total_tokens_all += result_dict.get('total_tokens', 0)
                total_cost_all += result_dict.get('total_cost', 0)

                # Write per-problem result to JSONL incrementally
                jsonl_record = {
                    'dataset': args.dataset,
                    'model': args.model,
                    'algorithm': args.algorithm,
                    'problem': result_dict['problem'],
                    'result': result_name,
                    'ground_truth': result_dict.get('ground_truth'),
                    'correct': result_name == 'ACCEPT',
                    'elapsed_time': result_dict.get('elapsed_time', 0),
                    'api_call_count': result_dict.get('api_call_count', 0),
                    'total_tokens': result_dict.get('total_tokens', 0),
                    'prompt_tokens': result_dict.get('prompt_tokens', 0),
                    'completion_tokens': result_dict.get('completion_tokens', 0),
                    'total_cost': result_dict.get('total_cost', 0),
                    'error': result_dict.get('error'),
                }
                with open(output_path, 'a', encoding='utf-8') as out_f:
                    out_f.write(json.dumps(jsonl_record) + '\n')

                pbar.update()
                pbar.set_description(
                    f'Acc: {accept_num}/{current_num} | '
                    f'WA: {wrong_answer_num} | '
                    f'CE: {compile_error_num} | '
                    f'RE: {runtime_error_num}'
                )

    # Print final summary with all four metrics
    print('\n' + '=' * 60)
    print(f'RESULTS SUMMARY: {args.dataset} | {args.model} | {args.algorithm}')
    print('=' * 60)
    print(f'Total problems:    {total_num}')
    print(f'Accept:            {accept_num}/{total_num} ({accept_num / total_num * 100:.2f}%)')
    print(f'Wrong Answer:      {wrong_answer_num}/{total_num} ({wrong_answer_num / total_num * 100:.2f}%)')
    print(f'Compile Error:     {compile_error_num}/{total_num} ({compile_error_num / total_num * 100:.2f}%)')
    print(f'Runtime Error:     {runtime_error_num}/{total_num} ({runtime_error_num / total_num * 100:.2f}%)')
    print(f'Total time:        {total_elapsed:.1f}s')
    print(f'Total tokens:      {total_tokens_all}')
    print(f'Total cost:        ${total_cost_all:.4f}')
    print(f'Results saved to:  {output_path}')
    print('=' * 60)

    # Also write a summary JSON file
    summary_path = output_path.replace('.jsonl', '_summary.json')
    summary = {
        'dataset': args.dataset,
        'model': args.model,
        'algorithm': args.algorithm,
        'total_problems': total_num,
        'max_problems': args.max_problems,
        'accept': accept_num,
        'wrong_answer': wrong_answer_num,
        'compile_error': compile_error_num,
        'runtime_error': runtime_error_num,
        'accuracy': accept_num / total_num * 100 if total_num > 0 else 0,
        'wrong_answer_rate': wrong_answer_num / total_num * 100 if total_num > 0 else 0,
        'compile_error_rate': compile_error_num / total_num * 100 if total_num > 0 else 0,
        'runtime_error_rate': runtime_error_num / total_num * 100 if total_num > 0 else 0,
        'total_elapsed_time': total_elapsed,
        'total_tokens': total_tokens_all,
        'total_cost': total_cost_all,
        'enable_reflection': args.enable_reflection,
        'max_collaborate_nums': args.max_collaborate_nums,
        'max_trials': args.max_trials,
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'Summary saved to:  {summary_path}')


if __name__ == '__main__':
    main()

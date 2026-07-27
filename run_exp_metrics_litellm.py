import argparse
import json
import time
import os
import re
import shutil
import subprocess
import traceback
from datetime import datetime, timezone
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


def load_latest_results(output_path):
    """Return the last valid JSONL record for each problem."""
    latest = {}
    if not os.path.exists(output_path):
        return latest
    with open(output_path, 'r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f'Invalid JSONL at {output_path}:{line_number}: {exc}'
                ) from exc
            problem = record.get('problem')
            if problem is not None:
                latest[str(problem)] = record
    return latest


def is_retryable_error(error):
    """Classify provider/network failures that should not become benchmark results."""
    if not error:
        return False
    message = str(error).lower()
    retryable_markers = (
        'ratelimiterror',
        'rate limit',
        'error code: 429',
        'timeout',
        'timed out',
        'apiconnectionerror',
        'connection error',
        'error code: 502',
        'error code: 503',
        'error code: 504',
    )
    return any(marker in message for marker in retryable_markers)


def is_retryable_record(record):
    if record.get('retryable') is True:
        return True
    return is_retryable_error(record.get('error'))


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
    problem_root = Path(path) / 'problems' / problem
    problem_artifact_dir = problem_root

    try:
        # A cancelled worker may leave an empty problem directory. Reuse that
        # empty directory; only allocate an attempt directory when artifacts
        # from an earlier attempt actually exist.
        problem_root.mkdir(parents=True, exist_ok=True)
        if any(problem_root.iterdir()):
            problem_artifact_dir = (
                problem_root
                / 'attempts'
                / datetime.now(timezone.utc).strftime(
                    'attempt_%Y%m%dT%H%M%S_%fZ'
                )
            )
            problem_artifact_dir.mkdir(parents=True, exist_ok=False)

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
                    max_trials=max_trials,
                    artifact_dir=str(problem_artifact_dir))
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

        final_dir = problem_artifact_dir / 'final'
        final_dir.mkdir(parents=True, exist_ok=True)

        # Canonical, readable per-problem artifacts.
        original_answer_path = final_dir / 'original_answer.txt'
        original_answer_path.write_text(answer, encoding='utf-8')

        code = extract_code_from_string(answer)
        code_file_path = final_dir / 'generated_code.py'
        code_file_path.write_text(code, encoding='utf-8')

        # Evaluate the canonical final program.
        ground_truth = get_ground_truth(dataset, problem)
        test_log_path = final_dir / 'test_log.txt'
        with test_log_path.open('w', encoding='utf-8') as handle:
            result = test_generated_code(
                problem,
                ground_truth,
                handle,
                generated_code_path=str(code_file_path),
            )

        # Compatibility copies preserve the previous layout for existing tools.
        legacy_original = Path(path) / f'{problem}_original_answer.txt'
        legacy_code = Path(path) / f'{problem}_generated_code.py'
        legacy_test_log = Path(path) / f'{problem}_test_log.txt'
        shutil.copy2(original_answer_path, legacy_original)
        shutil.copy2(code_file_path, legacy_code)
        shutil.copy2(test_log_path, legacy_test_log)
        compatibility_code = Path(path) / 'codes' / problem / 'generated_code.py'
        compatibility_code.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(code_file_path, compatibility_code)

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
            'artifact_dir': str(problem_artifact_dir),
            'generated_code_path': str(code_file_path),
            'test_log_path': str(test_log_path),
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        retryable = is_retryable_error(f'{type(e).__name__}: {e}')
        print(f"Error processing {problem}: {str(e)}")
        traceback.print_exc()
        trace = traceback.format_exc()
        try:
            problem_artifact_dir.mkdir(parents=True, exist_ok=True)
            (problem_artifact_dir / 'traceback.txt').write_text(
                trace,
                encoding='utf-8',
            )
            with (problem_artifact_dir / 'failure.json').open(
                'w',
                encoding='utf-8',
            ) as handle:
                json.dump({
                    'problem': problem,
                    'error_type': type(e).__name__,
                    'error': str(e),
                    'elapsed_time': elapsed_time,
                    'retryable': retryable,
                }, handle, indent=2)
        except OSError as artifact_error:
            print(
                f'Could not write failure artifact for {problem}: '
                f'{artifact_error}'
            )
        return {
            'problem': problem,
            'result': 'RUNTIME_ERROR',
            'ground_truth': None,
            'success': False,
            'error': str(e),
            'retryable': retryable,
            'failure_stage': 'pipeline_exception',
            'elapsed_time': elapsed_time,
            'api_call_count': 0,
            'total_tokens': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_cost': 0.0,
            'artifact_dir': str(problem_artifact_dir),
            'traceback_path': str(problem_artifact_dir / 'traceback.txt'),
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
    parser.add_argument('--run_dir', type=str, default=None, help='New, non-existing directory for this run')
    parser.add_argument('--resume_dir', type=str, default=None, help='Resume from an existing log directory')
    parser.add_argument('--output', type=str, default=None, help='Output JSONL file path for per-problem results')
    parser.add_argument('--run_index', type=int, default=None, help='Independent-run label stored in metadata/results')
    parser.add_argument('--provider', choices=('openai', 'openrouter'), default=None,
                        help='Provider label stored in metadata/results')
    args = parser.parse_args()
    args.algorithm = args.algorithm.lower()
    if args.run_dir and args.resume_dir:
        parser.error('--run_dir and --resume_dir are mutually exclusive')

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

    selected_problems = list(matched_problems)
    total_num = len(selected_problems)
    print(f'Running {total_num} problems from dataset {args.dataset}')

    # Determine log directory
    if args.resume_dir:
        path = os.path.abspath(args.resume_dir)
        if not os.path.isdir(path):
            print(f'Resume directory does not exist: {path}')
            exit(1)
        print(f'Resuming from {path}')
    else:
        if args.run_dir:
            path = os.path.abspath(args.run_dir)
        else:
            safe_model = re.sub(r'[^A-Za-z0-9_.-]+', '_', args.model)
            run_name = 'run_{}_{}_{}'.format(
                datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'),
                os.getpid(),
                time.time_ns() % 1_000_000_000,
            )
            path = os.path.abspath(os.path.join(
                args.log_dir,
                'reruns',
                args.dataset,
                safe_model,
                args.algorithm,
                run_name,
            ))
        Path(path).mkdir(parents=True, exist_ok=False)

    print(f'Save log to {path}')

    # Set up output JSONL path
    if args.output is None:
        output_path = os.path.join(path, 'results.jsonl')
    else:
        output_path = os.path.abspath(args.output)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    existing_results = load_latest_results(output_path) if args.resume_dir else {}
    already_recorded = {
        problem for problem, record in existing_results.items()
        if problem in selected_problems and not is_retryable_record(record)
    }
    matched_problems = [
        problem for problem in selected_problems
        if problem not in already_recorded
    ]
    if args.resume_dir:
        print(f'  Already recorded: {len(already_recorded)} problems')
        print(f'  Remaining: {len(matched_problems)} problems')

    try:
        source_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = None
    metadata = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'source_commit': source_commit,
        'run_dir': path,
        'output': output_path,
        'dataset': args.dataset,
        'model': args.model,
        'provider': args.provider,
        'algorithm': args.algorithm,
        'problem': args.problem,
        'max_problems': args.max_problems,
        'num_processes': args.num_processes,
        'enable_reflection': args.enable_reflection,
        'max_collaborate_nums': args.max_collaborate_nums,
        'max_trials': args.max_trials,
        'run_index': args.run_index,
        'artifact_layout': 'per_problem_v1',
    }
    if args.resume_dir:
        metadata_name = 'metadata_resume_{}.json'.format(
            datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        )
    else:
        metadata_name = 'metadata.json'
    with open(os.path.join(path, metadata_name), 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)

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
    num_processes = args.num_processes or min(50, max(1, len(matched_problems)))
    result_counts = {
        'ACCEPT': 0,
        'WRONG_ANSWER': 0,
        'COMPILE_ERROR': 0,
        'RUNTIME_ERROR': 0,
    }
    total_elapsed = 0.0
    total_tokens_all = 0
    total_cost_all = 0.0
    for record in existing_results.values():
        if str(record.get('problem')) not in already_recorded:
            continue
        result_name = record.get('result')
        if result_name in result_counts:
            result_counts[result_name] += 1
        total_elapsed += record.get('elapsed_time', 0) or 0
        total_tokens_all += record.get('total_tokens', 0) or 0
        total_cost_all += record.get('total_cost', 0) or 0
    accept_num = result_counts['ACCEPT']
    wrong_answer_num = result_counts['WRONG_ANSWER']
    compile_error_num = result_counts['COMPILE_ERROR']
    runtime_error_num = result_counts['RUNTIME_ERROR']
    current_num = len(already_recorded)

    # A fresh run must never overwrite an existing result file.
    output_mode = 'a' if args.resume_dir else 'x'
    with open(output_path, output_mode, encoding='utf-8'):
        pass

    print(f"Running with {num_processes} threads...")
    print(f"Results will be saved to: {output_path}")

    if matched_problems:
        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            futures = {executor.submit(process_problem, pa): pa for pa in problem_args}
            with tqdm(total=len(matched_problems)) as pbar:
                for future in as_completed(futures):
                    try:
                        result_dict = future.result()
                    except Exception as exc:
                        # Keep one unexpected worker failure from terminating
                        # every other in-flight problem in the run.
                        failed_args = futures[future]
                        print(
                            f"Worker crashed for {failed_args['problem']}: "
                            f'{exc}'
                        )
                        traceback.print_exc()
                        result_dict = {
                            'problem': failed_args['problem'],
                            'result': 'RUNTIME_ERROR',
                            'ground_truth': None,
                            'success': False,
                            'error': f'{type(exc).__name__}: {exc}',
                            'retryable': True,
                            'failure_stage': 'worker_crash',
                            'elapsed_time': 0,
                            'api_call_count': 0,
                            'total_tokens': 0,
                            'prompt_tokens': 0,
                            'completion_tokens': 0,
                            'total_cost': 0.0,
                        }
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

                    # Write per-problem result to JSONL incrementally.
                    jsonl_record = {
                        'dataset': args.dataset,
                        'model': args.model,
                        'provider': args.provider,
                        'algorithm': args.algorithm,
                        'run_index': args.run_index,
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
                        'success': result_dict.get('success'),
                        'retryable': result_dict.get('retryable', False),
                        'failure_stage': result_dict.get('failure_stage'),
                        'artifact_dir': result_dict.get('artifact_dir'),
                        'generated_code_path': result_dict.get('generated_code_path'),
                        'test_log_path': result_dict.get('test_log_path'),
                        'traceback_path': result_dict.get('traceback_path'),
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
    else:
        print('All selected problems already have JSONL records; rebuilding summary only.')

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
    final_records = load_latest_results(output_path)
    recorded_problems = sorted(
        {
            problem for problem, record in final_records.items()
            if problem in selected_problems and not is_retryable_record(record)
        },
        key=sort_key,
    )
    missing_problems = [
        problem for problem in selected_problems
        if problem not in recorded_problems
    ]
    summary_path = output_path.replace('.jsonl', '_summary.json')
    summary = {
        'dataset': args.dataset,
        'model': args.model,
        'provider': args.provider,
        'algorithm': args.algorithm,
        'run_index': args.run_index,
        'total_problems': total_num,
        'recorded_problems': len(recorded_problems),
        'missing_problems': missing_problems,
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

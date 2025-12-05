import os
import json
import time
import numpy as np
from pathlib import Path
from comment import Comment
from conductor import Conductor
from reducer import Reducer
from evaluator import Evaluator
from experts import (
    ModelingExpert,
    ProgrammingExpert,
    LPFileGenerator,
    ModelingKnowledgeSupplementExpert,
    ParameterExtractor,
    CodeReviewer,
    ProgrammingExampleProvider,
    TerminologyInterpreter,
)
from comment_pool import CommentPool
from utils import extract_code_from_string


def chain_of_experts(problem,
                     max_collaborate_nums,
                     model_name,
                     enable_reflection,
                     max_trials,
                     problem_name=None,
                     log_dir='log'):
    """Run Chain of Experts pipeline

    Args:
        problem: a dict of problem_description and code_example.
        problem_name: name of the problem (used for logging)
        log_dir: directory for logging generated code (only used if reflection is enabled)

    Return:
        code: code of problem
    """
    all_experts = [
        TerminologyInterpreter(model_name),
        ParameterExtractor(model_name),
        ModelingExpert(model_name),
        ProgrammingExampleProvider(model_name),
        ProgrammingExpert(model_name),
        # LPFileGenerator(model_name),
        ModelingKnowledgeSupplementExpert(model_name),
        CodeReviewer(model_name),
    ]
    num_experts = len(all_experts)
    reducer = Reducer(model_name)
    comment_pool = CommentPool(all_experts, visible_matrix=np.ones((num_experts, num_experts)))
    conductor = Conductor(model_name)
    evaluator = Evaluator(model_name)
    expert_stack = []

    # Create log directory if reflection is enabled
    code_file_path = None
    if enable_reflection:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_dir_name = f'run_reflection_{str(round(time.time()))}'
        code_root_dir = os.path.join(log_dir, log_dir_name)
        Path(code_root_dir).mkdir(parents=True, exist_ok=True)
        # Create problem-specific code directory (codes/{problem_name}/)
        if problem_name:
            problem_code_dir = os.path.join(code_root_dir, 'codes', problem_name)
            Path(problem_code_dir).mkdir(parents=True, exist_ok=True)
            code_file_path = os.path.join(problem_code_dir, 'generated_code.py')
        else:
            # Fallback: use root if problem_name not provided
            code_file_path = os.path.join(code_root_dir, 'generated_code.py')

    for _ in range(max_trials):
        for _ in range(max_collaborate_nums):
            next_expert = conductor.forward(problem, comment_pool, max_collaborate_nums)
            print(f'Choose next expert: {next_expert.name}')
            comment_text = next_expert.forward(problem, comment_pool)
            print(f'Given comment:\n{comment_text}')
            comment_pool.add_comment(Comment(next_expert, comment_text))
            expert_stack.append(next_expert)
        answer = reducer.forward(problem, comment_pool)

        if enable_reflection:
            # Write generated_code.py to log directory (never to root folder)
            code = extract_code_from_string(answer)
            with open(code_file_path, 'w') as f:
                f.write(code)
            test_sample = {"input": {}}
            print(f'Generate test sample:\n{test_sample}')
            test_samples = [test_sample]
            feedback = evaluator.evaluate(test_samples, generated_code_path=code_file_path)
            feedback_pool = CommentPool(all_experts, visible_matrix=np.ones((num_experts, num_experts)))
            feedback_pool.add_comment(Comment(evaluator, feedback))
            if feedback is not None:
                while expert_stack:
                    previous_expert = expert_stack.pop()
                    previous_comment = comment_pool.pop_comment()
                    result = previous_expert.backward(feedback_pool)

                    if result['is_caused_by_you']:
                        previous_comment.comment_text = result['refined_result']
                        expert_stack.append(previous_expert)
                        comment_pool.add_comment(previous_comment)
                        break
                    else:
                        feedback_pool.add_comment(Comment(previous_expert, result['reason']))
            else:
                break
    return answer


if __name__ == '__main__':
    from utils import read_problem
    problem = read_problem('LPWP', 'prob_250')
    chain_of_experts(problem, model_name='gpt-3.5-turbo-1106', enable_reflection=False)

import os
import json
import numpy as np
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
from pathlib import Path
import time


def _write_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(value), encoding='utf-8')


def chain_of_experts(problem, 
                     max_collaborate_nums, 
                     model_name, 
                     enable_reflection,
                     max_trials,
                     log_dir='log',
                     artifact_dir=None):
    """Run Chain of Experts pipeline
    
    Args:
        problem: a dict of problem_description and code_example.
    
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
    reflection_root = None
    if enable_reflection:
        if artifact_dir is not None:
            reflection_root = Path(artifact_dir) / 'reflection'
        else:
            reflection_root = Path(log_dir) / (
                f'run_reflection_{time.time_ns()}_{os.getpid()}'
            )
        reflection_root.mkdir(parents=True, exist_ok=True)

    for trial_index in range(max_trials):
        trial_dir = None
        selected_experts_path = None
        if reflection_root is not None:
            trial_dir = reflection_root / f'trial_{trial_index + 1:02d}'
            trial_dir.mkdir(parents=True, exist_ok=False)
            selected_experts_path = trial_dir / 'selected_experts.jsonl'

        for collaboration_index in range(max_collaborate_nums):
            next_expert = conductor.forward(problem, comment_pool, max_collaborate_nums)
            print(f'Choose next expert: {next_expert.name}')
            comment_text = next_expert.forward(problem, comment_pool)
            print(f'Given comment:\n{comment_text}')
            comment_pool.add_comment(Comment(next_expert, comment_text))
            expert_stack.append(next_expert)
            if selected_experts_path is not None:
                with selected_experts_path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps({
                        'collaboration': collaboration_index + 1,
                        'expert': next_expert.name,
                        'comment': comment_text,
                    }, default=str) + '\n')
        answer = reducer.forward(problem, comment_pool)
        if trial_dir is not None:
            _write_text(trial_dir / 'reducer_answer.txt', answer)

        code = extract_code_from_string(answer)
        code_file_path = None
        if trial_dir is not None:
            code_file_path = str(trial_dir / 'generated_code.py')
            _write_text(code_file_path, code)

        if enable_reflection:
            # test_sample = evaluator.forward(problem)
            test_sample = {"input": {}}
            print(f'Generate test sample:\n{test_sample}')
            test_samples = [test_sample]
            feedback = evaluator.evaluate(test_samples, generated_code_path=code_file_path)
            _write_text(
                trial_dir / 'evaluation_feedback.txt',
                '' if feedback is None else feedback,
            )
            feedback_pool = CommentPool(all_experts, visible_matrix=np.ones((num_experts, num_experts)))
            feedback_pool.add_comment(Comment(evaluator, feedback))
            if feedback is not None:
                backward_index = 0
                while expert_stack:
                    backward_index += 1
                    previous_expert = expert_stack.pop()
                    previous_comment = comment_pool.pop_comment()
                    raw_result = previous_expert.backward(feedback_pool)
                    backward_dir = (
                        trial_dir
                        / 'backward'
                        / f'step_{backward_index:02d}_{previous_expert.name}'
                    )
                    _write_text(backward_dir / 'raw_response.txt', raw_result)
                    parsed_text = raw_result.strip("```json").strip("```")
                    result = json.loads(parsed_text)
                    _write_text(
                        backward_dir / 'parsed_response.json',
                        json.dumps(result, indent=2, default=str),
                    )
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
    problem = read_problem('BWOR', '1')
    # chain_of_experts(problem, model_name='gpt-3.5-turbo-1106', enable_reflection=False)
    chain_of_experts(problem, model_name='o4-mini', enable_reflection=True, max_collaborate_nums=3, max_trials=3)

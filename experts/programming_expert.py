from experts.base_expert import BaseExpert

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class ProgrammingExpertBackwardResponse(BaseModel):
    """JSON schema for programming expert backward output"""
    is_caused_by_you: bool = Field(description="Whether the issue is caused by this expert")
    reason: str = Field(description="Explanation of the issue. Empty string if not caused by you")
    refined_result: str = Field(description="Refined code if the issue is caused by you")


class ProgrammingExpert(BaseExpert):

    ROLE_DESCRIPTION = 'You are a Python programmer in the field of operations research and optimization. Your proficiency in utilizing third-party libraries such as Gurobi is essential. In addition to your expertise in Gurobi, it would be great if you could also provide some background in related libraries or tools, like NumPy, SciPy, or PuLP.'
    FORWARD_TASK = '''You are given a specific problem. You aim to develop an efficient Python program that addresses the given problem.
Now the origin problem is as follow:
{problem_description}
Let's analyse the problem step by step, and then give your Python code.
Here is a starter code:
{code_example}
And the comments from other experts are as follow:
{comments_text}

IMPORTANT: The function must take NO arguments. All data should be hardcoded inside the function body.
IMPORTANT: The function should return ONLY the objective value, not the decision variables.

Give your Python code directly. You should follow the format of given code example strictly. No code is required outside the function except for the import package (No test code). In your code, the model must be a solvable LP or MIP model.'''
    BACKWARD_TASK = '''When you are solving a problem, you get a feedback from the external environment. You need to judge whether this is a problem caused by you or by other experts (other experts have given some results before you). If it is your problem, you need to give Come up with solutions and refined code.

The original problem is as follow:
{problem_description}

The code you give previously is as follow:
{previous_code}
    
The feedback is as follow:
{feedback}

IMPORTANT: The function must take NO arguments. All data should be hardcoded inside the function body.
IMPORTANT: The function should return ONLY the objective value, not the decision variables.
'''

    def __init__(self, model):
        super().__init__(
            name='Programming Expert',
            description='Skilled in programming and coding, capable of implementing the optimization solution in a programming language.',
            model=model
        )
        self.backward_prompt_template = self.ROLE_DESCRIPTION + '\n' + self.BACKWARD_TASK
        # Create a separate LLM instance with structured output for backward chain
        backward_llm = ChatOpenAI(
            model_name=model,
            temperature=1.0
        )
        structured_backward_llm = backward_llm.with_structured_output(ProgrammingExpertBackwardResponse)
        prompt = PromptTemplate.from_template(self.backward_prompt_template)
        self.backward_chain = prompt | structured_backward_llm

    def forward(self, problem, comment_pool):
        self.problem = problem
        comments_text = comment_pool.get_current_comment_text()
        print('Input')
        print(self.FORWARD_TASK.format(
            problem_description=problem['description'],
            code_example=problem['code_example'],
            comments_text=comments_text
        ))
        print()
        result = self.forward_chain.invoke({
            'problem_description': problem['description'],
            'code_example': problem['code_example'],
            'comments_text': comments_text
        })
        output = result.content
        self.previous_code = output
        return output

    def backward(self, feedback_pool):
        if not hasattr(self, 'problem'):
            raise NotImplementedError('Please call forward first!')
        output = self.backward_chain.invoke(
            {
                "problem_description": self.problem['description'],
                "previous_code": self.previous_code,
                "feedback": feedback_pool.get_current_comment_text()
            }
        )
        return output.model_dump()

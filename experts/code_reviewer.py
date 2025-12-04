from experts.base_expert import BaseExpert

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
# from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional


class CodeReviewResponse(BaseModel):
    """JSON schema for code reviewer backward output"""
    is_caused_by_you: bool = Field(description="Whether the issue is caused by this expert")
    reason: str = Field(description="Explanation of the issue. Empty string if not caused by you")
    refined_result: str = Field(description="Refined code or answer if the issue is caused by you")


class CodeReviewer(BaseExpert):

    ROLE_DESCRIPTION = 'You are a code reviewer that conducts thorough reviews of the implemented code to identify any errors, inefficiencies, or areas for improvement.'
    FORWARD_TASK = '''As a Code Reviewer, your responsibility is to conduct thorough reviews of implemented code related to optimization problems.
You will identify possible errors, inefficiencies, or areas for improvement in the code, ensuring that it adheres to best practices and delivers optimal results. Now, here is the problem:
{problem_description}.

You are supposed to refer to the comments given by your colleagues from other aspects: {comments_text}

IMPORTANT: Do not change the function name in your output. Keep all function names exactly as they were in the original code.
IMPORTANT: The function must take NO arguments. All data should be hardcoded inside the function body.
IMPORTANT: The function should return ONLY the objective value, not the decision variables.'''

    BACKWARD_TASK = '''When you are solving a problem, you get a feedback from the external environment. You need to judge whether this is a problem caused by you or by other experts (other experts have given some results before you). If it is your problem, you need to give Come up with solutions and refined code.

The original problem is as follow:
{problem_description}

The answer you give previously is as follow:
{previous_answer}

The feedback is as follow:
{feedback}

IMPORTANT: Do not change the function name in your output. Keep all function names exactly as they were in the original code.
IMPORTANT: The function must take NO arguments. All data should be hardcoded inside the function body.
IMPORTANT: The function should return ONLY the objective value, not the decision variables.
'''

    def __init__(self, model):
        super().__init__(
            name='Code Reviewer',
            description='Skilled in programming and coding, capable of implementing the optimization solution in a programming language.',
            model=model
        )
        self.forward_prompt_template = self.ROLE_DESCRIPTION + '\n' + self.FORWARD_TASK
        self.forward_chain = PromptTemplate.from_template(self.forward_prompt_template) | self.llm
        self.backward_prompt_template = self.ROLE_DESCRIPTION + '\n' + self.BACKWARD_TASK
        # Create a separate LLM instance with structured output for backward chain
        backward_llm = ChatOpenAI(
            model_name=model,
            temperature=1.0
        )
        structured_backward_llm = backward_llm.with_structured_output(CodeReviewResponse)
        prompt = PromptTemplate.from_template(self.backward_prompt_template)
        self.backward_chain = prompt | structured_backward_llm

    def forward(self, problem, comment_pool):
        self.problem = problem
        comments_text = comment_pool.get_current_comment_text()
        result = self.forward_chain.invoke({
            'problem_description': problem['description'],
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
                "previous_answer": self.previous_code,
                "feedback": feedback_pool.get_current_comment_text()
            }
        )
        return output.model_dump()

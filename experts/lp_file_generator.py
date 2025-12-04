from experts.base_expert import BaseExpert

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class LPFileGeneratorBackwardResponse(BaseModel):
    """JSON schema for LP file generator backward output"""
    is_caused_by_you: bool = Field(description="Whether the issue is caused by this expert")
    reason: str = Field(description="Explanation of the issue. Empty string if not caused by you")
    refined_result: str = Field(description="Refined LP file if the issue is caused by you")


class LPFileGenerator(BaseExpert):

    ROLE_DESCRIPTION = 'You are an LP file generator that expertises in generating LP (Linear Programming) files that can be used by optimization solvers.'
    FORWARD_TASK = '''As an LP file generation expert, your role is to generate LP (Linear Programming) files based on the formulated optimization problem. 

LP files are commonly used by optimization solvers to find the optimal solution. 
Here is the important part source from LP file format document: {knowledge}. 

Your expertise in generating these files will help ensure compatibility and efficiency. 
Please review the problem description and the extracted information and provide the generated LP file: 
{problem_description}.

The comments given by your colleagues are as follows: 
{comments}, please refer to them carefully.'''

    BACKWARD_TASK = '''When you are solving a problem, you get a feedback from the external environment. You need to judge whether this is a problem caused by you or by other experts (other experts have given some results before you). If it is your problem, you need to give Come up with solutions and refined code.

The original problem is as follow:
{problem_description}

The feedback is as follow:
{feedback}

The modeling you give previously is as follow:
{previous_answer}
'''

    def __init__(self, model):
        super().__init__(
            name='LP File Generator',
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
        structured_backward_llm = backward_llm.with_structured_output(LPFileGeneratorBackwardResponse)
        prompt = PromptTemplate.from_template(self.backward_prompt_template)
        self.backward_chain = prompt | structured_backward_llm

    def forward(self, problem, comment_pool):
        self.problem = problem
        comments_text = comment_pool.get_current_comment_text()
        result = self.forward_chain.invoke({
            'problem_description': problem['description'],
            'knowledge': '',
            'comments': comments_text
        })
        output = result.content
        self.previous_model = output
        return output

    def backward(self, feedback_pool):
        if not hasattr(self, 'problem'):
            raise NotImplementedError('Please call forward first!')
        output = self.backward_chain.invoke(
            {
                "problem_description": self.problem['description'],
                "previous_answer": self.previous_answer,
                "feedback": feedback_pool.get_current_comment_text()
            }
        )
        return output.model_dump()

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI


class BaseExpert(object):

    def __init__(self, name, description, model, max_completion_tokens=None):
        self.name = name
        self.description = description
        self.model = model

        model_kwargs = {}
        if max_completion_tokens is not None:
            model_kwargs['max_completion_tokens'] = max_completion_tokens

        self.llm = ChatOpenAI(
            model_name=model,
            temperature=1.0,
            model_kwargs=model_kwargs
        )
        self.forward_prompt_template = self.ROLE_DESCRIPTION + '\n' + self.FORWARD_TASK
        self.forward_chain = PromptTemplate.from_template(self.forward_prompt_template) | self.llm
        if hasattr(self, 'BACKWARD_TASK'):
            self.backward_prompt_template = self.ROLE_DESCRIPTION + '\n' + self.BACKWARD_TASK
            self.backward_chain = PromptTemplate.from_template(self.backward_prompt_template) | self.llm

    def forward(self):
        pass

    def backward(self):
        pass

    def __str__(self):
        return f'{self.name}: {self.description}'

import os
from dotenv import load_dotenv
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI

load_dotenv()


class BaseExpert(object):

    def __init__(self, name, description, model):
        self.name = name
        self.description = description
        self.model = model
        kwargs = {}
        if any(x in self.model for x in ["gemini"]):
            kwargs.update({"extra_body":{"reasoning": {"effort": "high"}}})
        elif any(x in self.model for x in ["o3", "o4", "gpt-5"]):
            kwargs.update({"reasoning_effort": "high"})


        self.llm = ChatOpenAI(
            model_name=model,
            temperature=1.0,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_API_BASE"),
            **kwargs
        )
        self.forward_prompt_template = self.ROLE_DESCRIPTION + '\n' + self.FORWARD_TASK
        self.forward_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(self.forward_prompt_template)
        )
        if hasattr(self, 'BACKWARD_TASK'):
            self.backward_prompt_template = self.ROLE_DESCRIPTION + '\n' + self.BACKWARD_TASK
            self.backward_chain = LLMChain(
                llm=self.llm,
                prompt=PromptTemplate.from_template(self.backward_prompt_template)
            )

    def forward(self):
        pass

    def backward(self):
        pass

    def __str__(self):
        return f'{self.name}: {self.description}'

from model.llm_service import LLMService
from pydantic import BaseModel
from utils.nlp_logging import CustomLogger
import json

logger = CustomLogger(name="HuiRen qa generation api", write_to_file=True)
llm = LLMService(llm_logger=logger)


class KnowLedgePoint(BaseModel):
    id: str
    KnowledgePoint: str
    QuestionNum: str
    TagNum: str


class DataHelper:
    def __init__(self):
        self.prompt = self.get_prompt()

    def get_prompt(self):
        with open('./utils/prompt.json', encoding='utf-8') as config_file:
            return json.load(config_file)

    @staticmethod
    def get_messages(task_name, model_name, system_prompt, knowledge_point):
        if task_name == "tag_generation":
            system_prompt = system_prompt.replace('[tag_num]', knowledge_point.TagNum)
            print(system_prompt)
            prompt = f"生成特征标签的参考内容如下：\n{knowledge_point.KnowledgePoint}"
        else:
            prompt = f"生成特征标签的参考内容如下：\n{knowledge_point.KnowledgePoint}"

        if model_name != 'ERNIE-4.0-8K':
            messages = [{'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': prompt}, ]
        else:
            messages = [
                {'role': 'user', 'content': system_prompt + prompt}]
        return messages


def qa_generation(knowledge_point: KnowLedgePoint):
    data_helper = DataHelper()
    prompt = data_helper.prompt

    for model_name in ['gpt-4o', 'qwen-max', 'ERNIE-4.0-8K']:
        tag_prompt = prompt['qa_generation']['tag_generation']
        messages = data_helper.get_messages(task_name="tag_generation", model_name=model_name, system_prompt=tag_prompt,
                                            knowledge_point=knowledge_point)
        completion = llm.get_response(model_name=model_name, messages=messages)

        return completion

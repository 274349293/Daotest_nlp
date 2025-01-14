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
    QuestionType: str  # 1,2,3 填空，选择，问答


class DataHelper:
    def __init__(self):
        self.prompt = self.get_prompt()
        self.question_type_map = {"1": "choice_question_generation", "2": "fib_generation",
                                  "3": "short_answer_question_generation"}

    def get_prompt(self):
        with open('./utils/prompt.json', encoding='utf-8') as config_file:
            return json.load(config_file)

    @staticmethod
    def get_messages(task_name, model_name, system_prompt, knowledge_point, tags_list):
        if task_name == "tag_generation":
            system_prompt = system_prompt.replace('[tag_num]', knowledge_point.TagNum)
            prompt = f"生成特征标签的参考内容如下：\n{knowledge_point.KnowledgePoint}"
        elif task_name == "choice_question_generation" or "short_answer_question_generation":
            system_prompt = system_prompt.replace('[question_num]', knowledge_point.QuestionNum)
            prompt = f"本次学习材料的关键词list为：{tags_list}，\n\n 本次学习内容如下：\n{knowledge_point.KnowledgePoint}"
        else:
            # TODO
            pass

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
    tags_list = None
    for model_name in ['gpt-4o', 'qwen-max', 'ERNIE-4.0-8K']:
        tag_prompt = prompt['qa_generation']['tag_generation']
        tag_messages = data_helper.get_messages(task_name="tag_generation", model_name=model_name,
                                                system_prompt=tag_prompt,
                                                knowledge_point=knowledge_point, tags_list=tags_list)
        tags_list = json.loads(llm.get_response(model_name=model_name, messages=tag_messages))["result"]
        qa_prompt = prompt['qa_generation'][data_helper.question_type_map[knowledge_point.QuestionType]]
        qa_messages = data_helper.get_messages(task_name="choice_question_generation",
                                               model_name=model_name, system_prompt=qa_prompt,
                                               knowledge_point=knowledge_point, tags_list=tags_list)

        qa_res = llm.get_response(model_name=model_name, messages=qa_messages)
        print(qa_res)
        return qa_res

from model.llm_service import LLMService
from pydantic import BaseModel, Field
from utils.nlp_logging import CustomLogger
import json
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

logger = CustomLogger(name="DaoTest qa generation api", write_to_file=True)
llm = LLMService(llm_logger=logger)


def get_prompt():
    with open('./utils/prompt.json', encoding='utf-8') as config_file:
        return json.load(config_file)


class KnowledgePoint(BaseModel):
    questionNum: int
    additionalPrompt: str
    id: str
    knowledgePoint: str


def get_messages(model_name, system_prompt, kg_p):
    if model_name != 'ERNIE-4.0-8K':
        messages = [{'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': f"知识点内容如下：{kg_p.knowledgePoint}"}, ]
    else:
        messages = [
            {'role': 'user', 'content': system_prompt + f"知识点内容如下：{kg_p.knowledgePoint}"}]

    return messages


def decompose_knowledge_point(kg_p: KnowledgePoint):
    prompt = get_prompt()
    for model_name in ['gpt-4o', 'qwen-max', 'ERNIE-4.0-8K']:
        system_prompt = prompt['decompose_knowledge']
        messages = get_messages(model_name, system_prompt, kg_p)
        decompose_res = llm.get_response(model_name=model_name, messages=messages)
        print(decompose_res)

        return 1

from model.llm_service import LLMService
from pydantic import BaseModel
from utils.nlp_logging import CustomLogger
import json

logger = CustomLogger(name="DaoTest tag generation api", write_to_file=True)
llm = LLMService(llm_logger=logger)


class TagSet(BaseModel):
    id: str
    knowledgePoint: str
    tagList: list[str]
    tagNum: str


class DataHelper:
    def __init__(self):
        self.prompt = self.get_prompt()

    @staticmethod
    def get_prompt():
        with open('./utils/prompt.json', encoding='utf-8') as config_file:
            return json.load(config_file)

    @staticmethod
    def get_messages(model_name, system_prompt, tag_set, tags_list):

        system_prompt = system_prompt.replace('[tag_num]', tag_set.tagNum)
        if len(tags_list):
            system_prompt = system_prompt + f"这个list是人工为这段知识写的标签：{tag_set.tagList}，请参考这些标签的内容和形式来生成。"
        prompt = f"生成特征标签的参考内容如下：\n{tag_set.knowledgePoint}"

        if model_name != 'ERNIE-4.0-8K':
            messages = [{'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': prompt}, ]
        else:
            messages = [
                {'role': 'user', 'content': system_prompt + prompt}]
        logger.info("get messages success")
        return messages


def tag_generation(tag_set: TagSet):
    result = []
    try:
        data_helper = DataHelper()
        prompt = data_helper.prompt
        tags_list = tag_set.tagList
        for model_name in ['gpt-4o', 'qwen-max', 'ERNIE-4.0-8K']:
            tag_prompt = prompt['qa_generation']['tag_generation']
            tag_messages = data_helper.get_messages(model_name=model_name,
                                                    system_prompt=tag_prompt,
                                                    tag_set=tag_set, tags_list=tags_list)
            llm_res = json.loads(llm.get_response(model_name=model_name, messages=tag_messages))["result"]
            for tag_item in llm_res:
                if tag_item not in tags_list:
                    result.append(tag_item)
            logger.info(f"tag generation return success, model is {model_name}, result is {result}")
            return {"status": 1, "result": result}
    except Exception as e:
        logger.error(e)
        return {"status": 0, "result": result}

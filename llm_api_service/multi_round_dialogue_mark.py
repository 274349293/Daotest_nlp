from fastapi.responses import StreamingResponse
import ast
import json
from fastapi import FastAPI
from model.llm_service import LLMService
from pydantic import BaseModel
from utils.nlp_logging import CustomLogger

logger = CustomLogger(name="HuiRen mr dialogue mark api", write_to_file=True)
llm = LLMService(llm_logger=logger)


class DialogueMarkInfo(BaseModel):
    """
    system_prompt: "[{'role': 'system', 'content': system_prompt}]"
    history: "[{'role': 'system', 'content': system_prompt},{'role': 'user', 'content': 'user_input'},{'role': 'assistant', 'content': 'model_response'}]

    """
    id: str
    systemPrompt: str
    knowledge: str
    questionCase: str
    standardAnswer: str
    history: list


def get_mark_prompt(dialogue_mark_info: DialogueMarkInfo):
    if len(dialogue_mark_info.history) < 4:
        logger.error(f"There aren't enough rounds of dialogue, history len is {len(dialogue_mark_info.history)}")
        return None
    else:
        dialogue_history = ""
        for index, dialogue_item in enumerate(dialogue_mark_info.history):
            if index % 2 == 0:
                dialogue_history += f"assistant:{dialogue_item}\n"
            else:
                dialogue_history += f"user:{dialogue_item}\n"
        system_prompt = dialogue_mark_info.systemPrompt if len(dialogue_mark_info.systemPrompt) > 0 else " "
        knowledge = dialogue_mark_info.knowledge if len(dialogue_mark_info.knowledge) > 0 else " "
        case = dialogue_mark_info.questionCase if len(dialogue_mark_info.questionCase) > 0 else " "
        standard_answer = dialogue_mark_info.standardAnswer if len(dialogue_mark_info.standardAnswer) > 0 else " "

        messages = [{'role': 'system', 'content': system_prompt},
                    {'role': 'user',
                     'content': f"相关学习材料如下：\n{knowledge}\n案例如下：\n{case}\n标准答案如下：\n{standard_answer} 历史对话数据如下：{dialogue_history}"}]

        return messages


def json_formatting_repair(json_str: str):
    """
    1. 针对qwen,wenxin生成的回复带有 ```json``` 的case 进行修复
    :param json_str:
    :return:
    """

    # case1
    try:

        json_str = "{" + ''.join(json_str.split('{')[1:])
        json_str = ''.join(json_str.split('}')[0]) + "}"
        json_str = ast.literal_eval(json_str)

        return json_str
    except Exception as e:
        logger.error(f"json_formatting_repair case1:  {e}")
    return None


def multi_round_dialogue_mark(dialogue_mark_info: DialogueMarkInfo):
    logger.info("------------------start--------------------")

    message = get_mark_prompt(dialogue_mark_info=dialogue_mark_info)
    if message is not None:
        logger.info(f"get dialogue mark message success")
    else:
        logger.error(
            f"dialogue mark param error: system_prompt is {dialogue_mark_info.systemPrompt},"
            f" history is {dialogue_mark_info.history}")
        return "error"
    for model_name in ['gpt-4o', 'gpt-4o-mini', 'qwen-max']:
        try:
            completion = llm.get_response(model_name=model_name, messages=message)
            if completion is None:
                continue
            logger.info(f"{dialogue_mark_info.id} model name is {model_name} , get json result success")
            if model_name == 'gpt-4o' or model_name == 'gpt-4o-mini':
                response = json.loads(completion)
                logger.info(f"result is {response}")
                return response
            elif model_name == 'qwen-max':
                response = json_formatting_repair(completion)
                logger.info(f"result is {response}")
                return response
        except Exception as e:
            logger.error(f"mr dialogue mark error in {model_name}: {e}")
    logger.error(f"{dialogue_mark_info.id} dialogue mark api return failed , return error")
    return "error"

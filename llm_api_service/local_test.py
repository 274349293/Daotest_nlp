import ast
import json
from fastapi import FastAPI
from model.llm_service import LLMService
from pydantic import BaseModel
from utils.nlp_logging import CustomLogger

app = FastAPI()
logger = CustomLogger("HuiRen mr dialogue api")
llm = LLMService(llm_logger=logger)


class DialogueInfo(BaseModel):
    """
    system_prompt: "[{'role': 'system', 'content': system_prompt}]"
    history: "[{'role': 'system', 'content': system_prompt},{'role': 'user', 'content': 'user_input'},{'role': 'assistant', 'content': 'model_response'}]

    """
    id: str
    system_prompt: str
    history: str


def data_conversion(dialogue_info: DialogueInfo):
    """
    将入参的string格式的数据转为list
    :param string_data
    :return:list
    """
    history = []
    try:
        system_prompt = {'role': 'system', 'content': dialogue_info.system_prompt}
        history.append(system_prompt)
        for index, item in enumerate(ast.literal_eval(dialogue_info.history.replace('\n', ''))):
            if index % 2 == 0:
                history.append({'role': 'assistant', 'content': item})
            else:
                history.append({'role': 'user', 'content': item})
        return history
    except Exception as e:
        logger.error(f"data_conversion error:  {e}")
        return []


@app.post("/mr_dialogue")
def multi_round_dialogue(dialogue_info: DialogueInfo):
    def qwen_generate() -> bytes:
        for chunk in completion:
            try:
                chunk_content = f"data: {json.loads(chunk.model_dump_json())['choices'][0]['delta']['content']}\n\n"
                yield chunk_content
            except Exception as e:
                yield "data: [END]\n\n"
                logger.info(f"yield [END], qwen chunk is None, {e}")

    def chat_gpt_4o_generate() -> bytes:
        llm_resp = ""
        for chunk in completion:
            try:
                if (len(chunk.choices)) == 0:
                    continue
                if chunk.choices[0].delta.content == None:
                    llm_resp = str(llm_resp).replace('\n', '').replace('data: ', '')
                    logger.info(f"yield [END], chatgpt chunk is None, result is {llm_resp}")
                    yield "data: [END]\n\n"
                else:
                    chunk_content = f"data: {chunk.choices[0].delta.content}\n\n"
                    llm_resp = llm_resp + chunk_content

                    yield chunk_content

            except Exception as e:
                yield "data: [END]\n\n"
                logger.error(f"yield [END], chat_gpt_4o_generate fun error : {e}")

    logger.info("------------------start--------------------")

    _history = data_conversion(dialogue_info=dialogue_info)
    if len(_history):
        logger.info(f"data conversion success")
    else:
        logger.error(f"param error: system_prompt is {dialogue_info.system_prompt}, history is {dialogue_info.history}")
        return "error"
    for model_name in ['gpt-4o', 'qwen-max']:
        try:
            completion = llm.get_response(model_name=model_name, messages=_history)
            if completion is None:
                continue
            logger.info(f"{dialogue_info.id} model name is {model_name} , get stream completion success")
            if model_name == 'gpt-4o':
                a = json.loads(completion)
                return a
            elif model_name == 'qwen-max':
                a = json.loads(completion)
                return a
        except Exception as e:
            logger.error(f"mr dialogue error in {model_name}: {e}")
    logger.error(f"{dialogue_info.id} mr dialogue api return failed , return is error ")
    return "error"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8500)

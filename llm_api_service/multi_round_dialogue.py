from fastapi.responses import StreamingResponse
import json
from model.llm_service import LLMService
from pydantic import BaseModel
from utils.nlp_logging import CustomLogger

"""
智能培训项目2期任务

案例分析模块包含2个接口：
第一个接口-案例分析对话互动：
1.预设案例（案例内容固定，放在system prompt中）和题目，和用户进行多轮对话交互。
2.交互目的是让用户可以答对预设题目，后续轮次的对话意义在于引导用户将问题回答正确，对于用户回答错误的知识进行纠正并追问没有答全的知识点。
3.轮次暂定最多为4轮（用户回答最多4次），如果在4轮之前用户已经把知识点答全答好，则终止互动。
4.在用户回答完成后，给出对于用户这几次作答的评价和评分

第二个接口-案例分析对话内容评分和评价：
对第一个接口的对话内容进行评分和评价

update：
1.20241206 案例分析互动过程中，模型的回复字数不超过300字 
2.20241209 新增一个接口，二期任务中在视频中加的互动题，为行为风格测试（选择题），积分规则详见《建立信任关系的沟通》ppt
3.20241216 评价分数不合理，怎么答都是高分，只评价user回复部分
4.20241218 回复过程中模型的举例说明不能太详细（例子不要太详细）
5.20241218 视频互动的接口定制预设第二个问题
#TODO 评价显示了总token数不超过50的bad case
"""

logger = CustomLogger(name="HuiRen mr dialogue api", write_to_file=True)
llm = LLMService(llm_logger=logger)


class DialogueInfo(BaseModel):
    """
    system_prompt: "[{'role': 'system', 'content': system_prompt}]"
    history: "[{'role': 'system', 'content': system_prompt},{'role': 'user', 'content': 'user_input'},{'role': 'assistant', 'content': 'model_response'}]

    """
    id: str
    systemPrompt: str
    history: list


def data_conversion(dialogue_info: DialogueInfo):
    """
    将入参的string格式的数据转为list
    :param string_data
    :return:list
    """
    history = []
    try:
        system_prompt = {'role': 'system', 'content': dialogue_info.systemPrompt}
        history.append(system_prompt)
        for index, item in enumerate(dialogue_info.history):
            if index % 2 == 0:
                history.append({'role': 'assistant', 'content': item})
            else:
                history.append({'role': 'user', 'content': item})
        return history
    except Exception as e:
        logger.error(f"data_conversion error:  {e}")
        return []


async def multi_round_dialogue(dialogue_info: DialogueInfo):
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
        logger.info(f"multi round dialogue data conversion success")
    else:
        logger.error(
            f"multi round dialogue param error: system_prompt is {dialogue_info.systemPrompt}, "
            f"history is {dialogue_info.history}")
        return "error"
    for model_name in ['gpt-4o', 'gpt-4o-mini', 'qwen-max']:
        try:
            completion = llm.get_response_stream(model_name=model_name, messages=_history)
            if completion is None:
                continue
            logger.info(f"{dialogue_info.id} model name is {model_name} , get stream completion success")
            if model_name == 'gpt-4o' or model_name == 'gpt-4o-mini':
                return StreamingResponse(chat_gpt_4o_generate(), media_type="text/event-stream")
            elif model_name == 'qwen-max':
                return StreamingResponse(qwen_generate(), media_type="text/event-stream")
        except Exception as e:
            logger.error(f"mr dialogue error in {model_name}: {e}")
    logger.error(f"{dialogue_info.id} mr dialogue api return failed , return is error ")
    return "error"

from fastapi.responses import StreamingResponse
import ast
import json
from fastapi import FastAPI
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

app = FastAPI()
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


class BehavioralStyleInfo(BaseModel):
    A1: int
    B1: int
    A2: int
    B2: int
    A3: int
    B3: int
    A4: int
    B4: int
    A5: int
    B5: int
    A6: int
    B6: int
    A7: int
    B7: int
    A8: int
    B8: int
    A9: int
    B9: int
    A10: int
    B10: int
    A11: int
    B11: int
    A12: int
    B12: int
    A13: int
    B13: int
    A14: int
    B14: int
    A15: int
    B15: int
    A16: int
    B16: int
    A17: int
    B17: int
    A18: int
    B18: int


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


@app.post("/mr_dialogue_mark")
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


def behavioral_style_score_calculate(behavioral_style_info: BehavioralStyleInfo):
    o_score = (behavioral_style_info.A1 + behavioral_style_info.B3 + behavioral_style_info.A5 + behavioral_style_info.B7
               + behavioral_style_info.A9 + behavioral_style_info.B11 + behavioral_style_info.A13 +
               behavioral_style_info.B15 + behavioral_style_info.A17)
    s_score = (behavioral_style_info.B1 + behavioral_style_info.A3 + behavioral_style_info.B5 + behavioral_style_info.A7
               + behavioral_style_info.B9 + behavioral_style_info.A11 + behavioral_style_info.B13 +
               behavioral_style_info.A15 + behavioral_style_info.B17)
    d_score = (behavioral_style_info.B2 + behavioral_style_info.A4 + behavioral_style_info.B6 + behavioral_style_info.A8
               + behavioral_style_info.B10 + behavioral_style_info.A12 + behavioral_style_info.B14 +
               behavioral_style_info.A16 + behavioral_style_info.B18)
    i_score = (behavioral_style_info.A2 + behavioral_style_info.B4 + behavioral_style_info.A6 + behavioral_style_info.B8
               + behavioral_style_info.A10 + behavioral_style_info.B12 + behavioral_style_info.A14 +
               behavioral_style_info.B16 + behavioral_style_info.A18)
    if o_score > s_score and d_score > i_score:
        return "社交型"
    elif o_score > s_score and i_score > d_score:
        return "关系型"
    elif s_score > o_score and d_score > i_score:
        return "指导型"
    elif s_score > o_score and i_score > d_score:
        return "思考型"
    else:
        logger.error(f"behavioral style info is {behavioral_style_info}")
        return None


@app.post("/behavioral_style_test")
def behavioral_style(behavioral_style_info: BehavioralStyleInfo):
    logger.info("------------------start--------------------")
    logger.info(behavioral_style_info)
    try:
        result = behavioral_style_score_calculate(behavioral_style_info)
        if result is not None:
            logger.info(f"behavioral style test result is {result}")
            return {"result": result}
        else:
            logger.error(f"behavioral style test score calculate error, calculate fun return None")
            return "error"
    except Exception as e:
        logger.error(f"behavioral_style error: {e}")
        return "error"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8400)

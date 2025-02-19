import json
import ast
from model.llm_service import LLMService
from pydantic import BaseModel
from utils.nlp_logging import CustomLogger


def get_prompt():
    with open('./utils/prompt.json', encoding='utf-8') as prompt:
        return json.load(prompt)


logger = CustomLogger("HuiRen exam marking api")
system_prompt = get_prompt()["exam_marking"]

llm = LLMService(llm_logger=logger)


class ExamQaInfo(BaseModel):
    id: str
    question: str
    answer: str
    standardAnswer: str
    # passMusterAnswer: str
    # badAnswer: str
    qaTag: list
    unit: str
    # standardAnswerInfo: str
    # passMusterAnswerInfo: str
    # badAnswerInfo: str


def score_mapping(llm_result: dict):
    try:
        mapping_dict = {"低": 0, "中": 1, "高": 2}
        assert type(llm_result) is dict
        if llm_result["score"] not in mapping_dict:
            logger.error(f"llm socre error, llm score is {llm_result['score']}")
            llm_result["score"] = 1
        else:
            if llm_result["score"] == "低":
                llm_result["score"] = 0
            elif llm_result["score"] == "中":
                llm_result["score"] = 1
            elif llm_result["score"] == "高":
                llm_result["score"] = 2

    except Exception as e:
        llm_result["score"] = 1
        logger.error(f"llm socre mapping error: {e}")

    return llm_result


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


def exam_mark(qa_info: ExamQaInfo):
    logger.info("------------------start--------------------")
    for model_name in ['gpt-4o', 'qwen-max', 'ERNIE-4.0-8K']:
        try:
            rag_content = {"题目": qa_info.question, "标准答案": qa_info.standardAnswer,
                           "员工作答": qa_info.answer}

            if model_name != 'ERNIE-4.0-8K':
                messages = [{'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': f"以下是员工作答的题目和答案信息：{str(rag_content)}"}, ]
            else:
                messages = [
                    {'role': 'user', 'content': system_prompt + f"以下是员工作答的题目和答案信息：{str(rag_content)}"}]
            completion = llm.get_response(model_name=model_name, messages=messages)

            if model_name == 'gpt-4o':
                result = json.loads(completion)
                logger.info(f"exam mark api return {qa_info.id} success, model name is {model_name}")
                logger.info(f"result is  {result}")
                return result

            elif model_name == 'qwen-max' or model_name == 'ERNIE-4.0-8K':
                result = json_formatting_repair(completion)
                logger.info(f"exam mark api return {qa_info.id} success, model name is {model_name}")
                logger.info(f"result is  {result}")
                return result

        except Exception as e:
            logger.error(f"exam mark error in {model_name}: {e}")
    logger.error(f"{qa_info.id} exam mark api return failed , return is None")
    return {"score": 1, "llm_result": "None"}

import json
import ast
from fastapi import FastAPI
from model.llm_service import LLMService
from pydantic import BaseModel
from nlp_logging import CustomLogger

app = FastAPI()
logger = CustomLogger(name="HuiRen exam marking api", write_to_file=False)
system_prompt = """汇仁是一家大型医药企业集团，汇仁公司新招聘了一批新员工，公司已经对新员工完成了入职培训，现在以考试的方式检查新员工是否已经掌握了培训内容。你是一名资深的培训经理，你的任务如下：{
1.对员工的作答评分，评分规定：满分为100分，最低分为0分，请从0-100直接给出一个整数作为员工作答分数。先输出评价，再进行评分。
2.评分标准如下：
{
2.1 首先看标准答案中涵盖了几个知识点，如果涵盖了多个知识点，按照命中的标准答案中的知识点个数给分。这是一个标准答案的示例：{
1、制定开发目标
2、确定开发客户可以使用的政策
3、制定开发客户成功的标准
4、制定客户开发步骤
}，这条标准答案有4个知识点。
2.2 如果标准答案中涵盖2个及以上知识点，则每个知识点的分数平均分配。例如标准答案中如果有2个知识点，则每个知识点50分，标准答案中如果有4个知识点，则每个知识点25分。
2.3 如果标准答案中涵盖2个及以上知识点，员工的作答也涉及到多条知识点，并且员工的作答中涵盖的知识点个数少于等于标准答案的知识点，则按照员工命中了几个标准答案的知识点来给分。
2.4 如果标准答案中涵盖2个及以上知识点，员工的作答也涉及到多条知识点，并且员工的作答中涵盖的知识点个数多于标准答案的知识点，额外注意员工作答中是否写了无效的答案或者作答中多个知识点对应标准答案的某一个知识点的情况，按照员工命中了几个标准答案的知识点来给分。
2.5 如果标准答案中某个知识点是X分，而员工的作答中也写了该知识点，但是写的不全，或者理解有偏差，可以对该知识点打分时候打0-X分的任意正整数的分数。
2.6 如果标准答案中只有一条知识点，而员工的作答中不论写了几个知识点，只需要关注员工的作答是否完全覆盖了该条标准答案的知识点，和标准答案越相似分数越高。
2.7 如果员工的作答和标准答案差距非常大，或认定为是无效作答，则直接给0分。
}
3.评价内容为根据评分标准，对员工给出分数的理由
4.员工的评价是由语音转成文字作为输入，个别文字在转换期间可能会出现错别字情况和标点符号错误，标点符号丢失，标点符号错位等情况，如果遇到这些情况不要进行扣分。
5.回答的输出格式固定为json格式，示例如下：
{
    "llm_result": 对员工作答的评价,
    "score": 员工作答的评分
}

}
"""
llm = LLMService(llm_logger=logger)


class QaInfo(BaseModel):
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


@app.post("/exam_mark_mini")
def exam_mark_mini(qa_info: QaInfo):
    logger.info("------------------start--------------------")
    for model_name in ['gpt-4o-mini', 'qwen-max', 'ERNIE-4.0-8K']:
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
            print("??????????????????????")

            if model_name == 'gpt-4o-mini':
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


@app.post("/exam_mark_qwen")
def exam_mark_qwen(qa_info: QaInfo):
    logger.info("------------------start--------------------")
    for model_name in ['qwen-max', 'gpt-4o-mini', 'ERNIE-4.0-8K']:
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
            print("??????????????????????")

            if model_name == 'gpt-4o-mini':
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8101)

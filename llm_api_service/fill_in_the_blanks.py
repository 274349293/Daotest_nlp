from fastapi import FastAPI
import json
import ast
from pydantic import BaseModel
from model.llm_service import LLMService
from utils.nlp_logging import CustomLogger

logger = CustomLogger(name="HuiRen fib api")
system_prompt = """汇仁是一家大型医药企业集团，汇仁公司新招聘了一批新员工，公司已经对新员工完成了入职培训，现在以考试的方式检查新员工是否已经掌握了培训内容。你是一名资深的培训经理，你的任务如下：{
1.判断员工作答的对错。考试题目全部为填空题，题目中的每一个##代表一个所需要填写的空位，每个题目中如果有N个空位，则该题目所需要写N个答案，答案中每个空位的答案使用 | 隔开。示例如下：{
题目：在与客户洽谈时间的安排上，要保证时间满足##、##和##的要求。
标准答案：客户|公司审批|活动准备
}，这条题目需要填写3个空位，标准答案中有三个空位的答案依次对应题目中的空位，将标准答案填写到题目中则应该为：在与客户洽谈时间的安排上，要保证时间满足客户、公司审批和活动准备的要求。
2.判断对错的标准如下：
{
2.1 逐个比较员工作答每个空位的答案和标准答案中每个空位的答案，如果完全相同，则该空位标记为正确。如果不完全相同，但将答案填入题目中所表达的语义相同或95%以上相同，则该空位也标记为正确。
2.2 逐个比较员工作答每个空位的答案和标准答案中每个空位的答案，如果不完全相同，并且将答案填入题目中所表达的语义也不相同，则该空位标记为错误。
2.3 如果员工的作答和标准答案差距非常大，则该空位标记为错误。
2.4 用户作答中出现符号与标准答案不同的情况，如大小写不同，或者%写成“百分之”类似情况，只要语义和标准答案相同，一律判断为正确。
2.5 正确的填空为标记为1，错误的填空位标记为0。示例如下：{
题目：在与客户洽谈时间的安排上，要保证时间满足##、##和##的要求。
标准答案：客户|公司审批|活动准备
用户作答：自己|公司审批通过|活动准备
判断结果：[0,1,1]
}
}
3.回答的输出格式为固定格式，为json格式，示例如下：
{
    "result":[0,1,1]
}
"""
llm = LLMService(llm_logger=logger)
app = FastAPI()


class QaInfo(BaseModel):
    id: str
    question: str
    answer: str
    standardAnswer: str


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


@app.post("/fib")
def fill_in_the_blanks(qa_info: QaInfo):
    logger.info("------------------start--------------------")

    for model_name in ['gpt-4o', 'gpt-4o-mini', 'qwen-max', 'ERNIE-4.0-8K']:
        try:
            rag_content = {"题目": qa_info.question, "标准答案": qa_info.standardAnswer,
                           "员工作答": qa_info.answer}

            if model_name != 'ERNIE-4.0-8K':
                messages = [{'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': f"以下是员工作答的题目和答案信息：{str(rag_content)}"}, ]
            else:
                messages = [
                    {'role': 'user',
                     'content': system_prompt + f"以下是员工作答的题目和答案信息：{str(rag_content)}"}]
            completion = llm.get_response(model_name=model_name, messages=messages)

            if model_name == 'gpt-4o' or model_name == 'gpt-4o-mini':
                result = json.loads(completion)
                logger.info(f"fib api return {qa_info.id} success, model name is {model_name}")
                logger.info(f"result is  {result}")
                return result

            elif model_name == 'qwen-max' or model_name == 'ERNIE-4.0-8K':
                result = json_formatting_repair(completion)
                logger.info(f"fib api return {qa_info.id} success, model name is {model_name}")
                logger.info(f"result is  {result}")
                return result

        except Exception as e:
            logger.error(f"fib error in {model_name}: {e}")
    logger.error(f"{qa_info.id} fib api return failed , return is None")
    return {"score": 1, "llm_result": "None"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8200)

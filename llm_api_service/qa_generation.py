from model.llm_service import LLMService
from pydantic import BaseModel, Field
from utils.nlp_logging import CustomLogger
import json
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

logger = CustomLogger(name="DaoTest qa generation api", write_to_file=True)
llm = LLMService(llm_logger=logger)


class ChoiceQuestion(BaseModel):
    questionNum: int
    additionalPrompt: str


class FillInTheBlankQuestion(BaseModel):
    questionNum: int
    additionalPrompt: str


class ShortAnswerQuestion(BaseModel):
    questionNum: int
    additionalPrompt: str


class ReadingComprehensionQuestion(BaseModel):
    questionNum: int
    passage: str
    additionalPrompt: str


class CaseAnalysisQuestion(BaseModel):
    questionNum: int
    passage: str
    additionalPrompt: str


class QaGeneration(BaseModel):
    id: str
    type: int
    retryFlag: int
    knowledgeTitle: str
    knowledgePoint: str
    tagList: List[str] = Field(..., alias="tagList")
    choiceQuestion: Optional[ChoiceQuestion] = None
    fillInTheBlankQuestion: Optional[FillInTheBlankQuestion] = None
    shortAnswerQuestion: Optional[ShortAnswerQuestion] = None
    readingComprehensionQuestion: Optional[ReadingComprehensionQuestion] = None
    caseAnalysisQuestion: Optional[CaseAnalysisQuestion] = None


class DataHelper:
    def __init__(self):
        self.prompt = self.get_prompt()

    @staticmethod
    def get_prompt():
        with open('./utils/prompt.json', encoding='utf-8') as config_file:
            return json.load(config_file)

    @staticmethod
    def get_messages(task_name, model_name, system_prompt, qa_gen):
        if task_name == "tag_generation":
            system_prompt = system_prompt.replace('[tag_num]', "10")
            prompt = f"生成特征标签的参考内容如下：\n{qa_gen.knowledgePoint}"
        elif task_name == "decompose_knowledge_point":
            system_prompt = system_prompt
            prompt = f"本次学习材料的关键词list为：{str(qa_gen.tagList)}，\n\n 需要拆分的知识点如下：\n{qa_gen.knowledgePoint}"
        elif task_name == "choice_question_generation":
            system_prompt = system_prompt.replace('[question_num]', str(qa_gen.choiceQuestion.questionNum))
            additional_prompt = f"除了上述规则外会有额外的附加规则，如果和上面的规则有冲突，则执行上述规则，不执行额外附加规则，附加规则如下：{qa_gen.choiceQuestion.additionalPrompt} \n"
            prompt = additional_prompt + f"本次学习材料的关键词list为：{str(qa_gen.tagList)}，\n\n 本次学习内容如下：\n{qa_gen.knowledgePoint}"
        elif task_name == "short_answer_question_generation":
            system_prompt = system_prompt.replace('[question_num]', str(qa_gen.shortAnswerQuestion.questionNum))
            additional_prompt = f"除了上述规则外会有额外的附加规则，如果和上面的规则有冲突，则执行上述规则，不执行额外附加规则，附加规则如下：{qa_gen.shortAnswerQuestion.additionalPrompt} \n"
            prompt = additional_prompt + f"本次学习材料的关键词list为：{str(qa_gen.tagList)}，\n\n 本次学习内容如下：\n{qa_gen.knowledgePoint}"
        elif task_name == "fib_generation":
            system_prompt = system_prompt.replace('[question_num]', str(qa_gen.fillInTheBlankQuestion.questionNum))
            additional_prompt = f"除了上述规则外会有额外的附加规则，如果和上面的规则有冲突，则执行上述规则，不执行额外附加规则，附加规则如下：{qa_gen.fillInTheBlankQuestion.additionalPrompt} \n"
            prompt = additional_prompt + f"本次学习材料的关键词list为：{str(qa_gen.tagList)}，\n\n 本次学习内容如下：\n{qa_gen.knowledgePoint}"
        elif task_name == "reading_comprehension_question_generation":
            system_prompt = system_prompt.replace('[question_num]',
                                                  str(qa_gen.readingComprehensionQuestion.questionNum))
            additional_prompt = f"除了上述规则外会有额外的附加规则，如果和上面的规则有冲突，则执行上述规则，不执行额外附加规则，附加规则如下：{qa_gen.readingComprehensionQuestion.additionalPrompt} \n"
            prompt = additional_prompt + f"本次学习材料的关键词list为：{str(qa_gen.tagList)}，\n\n 本次学习的知识点内容如下：\n{qa_gen.knowledgePoint} \n\n本次的案例材料内容如下：{qa_gen.readingComprehensionQuestion.passage}"
        elif task_name == "case_analysis_question_generation":
            system_prompt = system_prompt.replace('[question_num]',
                                                  str(qa_gen.caseAnalysisQuestion.questionNum))
            additional_prompt = f"除了上述规则外会有额外的附加规则，如果和上面的规则有冲突，则执行上述规则，不执行额外附加规则，附加规则如下：{qa_gen.caseAnalysisQuestion.additionalPrompt} \n"
            prompt = additional_prompt + f"本次学习材料的关键词list为：{str(qa_gen.tagList)}，\n\n 本次学习的知识点内容如下：\n{qa_gen.knowledgePoint} \n\n本次的案例材料内容如下：{qa_gen.caseAnalysisQuestion.passage}"
        else:
            logger.error(f"task name error: {task_name} not in task dict")

            # 错误后默认使用选择题的任务prompt
            system_prompt = system_prompt.replace('[question_num]', str(qa_gen.choiceQuestion.questionNum))
            additional_prompt = f"除了上述规则外会有额外的附加规则，如果和上面的规则有冲突，则执行上述规则，不执行额外附加规则，附加规则如下：{qa_gen.choiceQuestion.additionalPrompt} \n"
            prompt = additional_prompt + f"本次学习材料的关键词list为：{str(qa_gen.tagList)}，\n\n 本次学习内容如下：\n{qa_gen.knowledgePoint}"

        if model_name != 'ERNIE-4.0-8K':
            messages = [{'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': prompt}, ]
        else:
            messages = [
                {'role': 'user', 'content': system_prompt + prompt}]
        return messages


def choice_question_generation(data_helper, qa_gen):
    choice_question_res = {"status": 0, "result": []}
    if qa_gen.choiceQuestion.questionNum == 0:
        logger.info("choice question num is 0")
        return None
    else:
        try:
            prompt = data_helper.prompt
            for model_name in ['gpt-4o', 'qwen-max', 'ERNIE-4.0-8K']:
                choice_question_prompt = prompt['qa_generation']['choice_question_generation']
                choice_question_messages = data_helper.get_messages(task_name="choice_question_generation",
                                                                    model_name=model_name,
                                                                    system_prompt=choice_question_prompt,
                                                                    qa_gen=qa_gen)

                choice_question_res["result"] = json.loads(
                    llm.get_response(model_name=model_name, messages=choice_question_messages))["result"]
                if qa_gen.choiceQuestion.questionNum != len(choice_question_res['result']):
                    logger.warning(
                        f"The number of choice questions generated is {len(choice_question_res['result'])}"
                        f" != parameters choiceQuestion.questionNum {qa_gen.choiceQuestion.questionNum}")
                    if len(choice_question_res['result']) > qa_gen.choiceQuestion.questionNum:
                        choice_question_res["result"] = choice_question_res['result'][
                                                        :qa_gen.choiceQuestion.questionNum]
                    else:
                        choice_question_res["result"] = (choice_question_res["result"] + json.loads(
                            llm.get_response(model_name=model_name, messages=choice_question_messages))["result"])[
                                                        :qa_gen.choiceQuestion.questionNum]
                logger.info(
                    f"choice question result were generated successfully, The number of generated is {qa_gen.choiceQuestion.questionNum}, model name is {model_name}")
                choice_question_res["status"] = 1
                return choice_question_res
        except Exception as e:
            logger.error(e)
            return choice_question_res


def fib_question_generation(data_helper, qa_gen):
    fib_question_res = {"status": 0, "result": []}
    if qa_gen.fillInTheBlankQuestion.questionNum == 0:
        logger.info("fib question num is 0")
        return None
    else:
        try:
            prompt = data_helper.prompt
            for model_name in ['gpt-4o', 'qwen-max', 'ERNIE-4.0-8K']:
                fib_question_prompt = prompt['qa_generation']['fib_generation']
                fib_question_messages = data_helper.get_messages(task_name="fib_generation",
                                                                 model_name=model_name,
                                                                 system_prompt=fib_question_prompt,
                                                                 qa_gen=qa_gen)

                fib_question_res["result"] = json.loads(
                    llm.get_response(model_name=model_name, messages=fib_question_messages))["result"]
                if qa_gen.fillInTheBlankQuestion.questionNum != len(fib_question_res['result']):
                    logger.warning(
                        f"The number of fib questions generated is {len(fib_question_res['result'])}"
                        f" != parameters fib_question.questionNum {qa_gen.fillInTheBlankQuestion.questionNum}")
                    if len(fib_question_res['result']) > qa_gen.fillInTheBlankQuestion.questionNum:
                        fib_question_res["result"] = fib_question_res['result'][
                                                     :qa_gen.fillInTheBlankQuestion.questionNum]
                    else:
                        fib_question_res["result"] = (fib_question_res["result"] + json.loads(
                            llm.get_response(model_name=model_name, messages=fib_question_res))["result"])[
                                                     :qa_gen.fillInTheBlankQuestion.questionNum]
                logger.info(
                    f"fib question result were generated successfully, The number of generated is {qa_gen.fillInTheBlankQuestion.questionNum}, model name is {model_name}")
                fib_question_res["status"] = 1
                return fib_question_res
        except Exception as e:
            logger.error(e)
            return fib_question_res


def short_answer_question_generation(data_helper, qa_gen):
    short_answer_question_res = {"status": 0, "result": []}
    if qa_gen.shortAnswerQuestion.questionNum == 0:
        logger.info("short answer question num is 0")
        return None
    else:
        try:
            prompt = data_helper.prompt
            for model_name in ['gpt-4o', 'qwen-max', 'ERNIE-4.0-8K']:
                short_answer_question_prompt = prompt['qa_generation']['short_answer_question_generation']
                short_answer_question_messages = data_helper.get_messages(task_name="short_answer_question_generation",
                                                                          model_name=model_name,
                                                                          system_prompt=short_answer_question_prompt,
                                                                          qa_gen=qa_gen)

                short_answer_question_res["result"] = json.loads(
                    llm.get_response(model_name=model_name, messages=short_answer_question_messages))["result"]
                if qa_gen.shortAnswerQuestion.questionNum != len(short_answer_question_res['result']):
                    logger.warning(
                        f"The number of short answer questions generated is {len(short_answer_question_res['result'])}"
                        f" != parameters short_answer_question.questionNum {qa_gen.shortAnswerQuestion.questionNum}")
                    if len(short_answer_question_res['result']) > qa_gen.shortAnswerQuestion.questionNum:
                        short_answer_question_res["result"] = short_answer_question_res['result'][
                                                              :qa_gen.shortAnswerQuestion.questionNum]
                    else:
                        short_answer_question_res["result"] = (short_answer_question_res["result"] + json.loads(
                            llm.get_response(model_name=model_name, messages=short_answer_question_res))["result"])[
                                                              :qa_gen.shortAnswerQuestion.questionNum]
                logger.info(
                    f"short answer question result were generated successfully, The number of generated is {qa_gen.shortAnswerQuestion.questionNum}, model name is {model_name}")
                short_answer_question_res["status"] = 1
                return short_answer_question_res
        except Exception as e:
            logger.error(e)
            return short_answer_question_res


def reading_comprehension_question_generation(data_helper, qa_gen):
    reading_comprehension_question_res = {"status": 0, "result": []}
    if qa_gen.readingComprehensionQuestion.questionNum == 0:
        logger.info("reading comprehension question num is 0")
        return None
    else:
        try:
            prompt = data_helper.prompt
            for model_name in ['gpt-4o', 'qwen-max', 'ERNIE-4.0-8K']:
                reading_comprehension_question_prompt = prompt['qa_generation'][
                    'reading_comprehension_question_generation']
                reading_comprehension_question_messages = data_helper.get_messages(
                    task_name="reading_comprehension_question_generation",
                    model_name=model_name,
                    system_prompt=reading_comprehension_question_prompt,
                    qa_gen=qa_gen)

                reading_comprehension_question_res["result"] = \
                    json.loads(
                        llm.get_response(model_name=model_name, messages=reading_comprehension_question_messages))[
                        "result"]
                if qa_gen.readingComprehensionQuestion.questionNum * 3 != len(
                        reading_comprehension_question_res['result']):
                    logger.warning(
                        f"The reading comprehension question generated is {len(reading_comprehension_question_res['result'])}"
                        f" != parameters readingComprehensionQuestion.questionNum {qa_gen.readingComprehensionQuestion.questionNum}")
                    if len(reading_comprehension_question_res[
                               'result']) > qa_gen.readingComprehensionQuestion.questionNum * 3:
                        reading_comprehension_question_res["result"] = reading_comprehension_question_res['result'][
                                                                       :qa_gen.readingComprehensionQuestion.questionNum * 3]
                    else:
                        reading_comprehension_question_res["result"] = (reading_comprehension_question_res["result"] +
                                                                        json.loads(
                                                                            llm.get_response(model_name=model_name,
                                                                                             messages=reading_comprehension_question_res))[
                                                                            "result"])[
                                                                       :qa_gen.readingComprehensionQuestion.questionNum * 3]
                logger.info(
                    f"reading comprehension question result were generated successfully, The number of generated is {qa_gen.readingComprehensionQuestion.questionNum}, model name is {model_name}")
                reading_comprehension_question_res["status"] = 1

                # tmp merge question and case

                for i in range(len(reading_comprehension_question_res['result'])):
                    reading_comprehension_question_res['result'][i][0] = \
                        qa_gen.readingComprehensionQuestion.passage + "\n" + \
                        reading_comprehension_question_res['result'][i][0]

                return reading_comprehension_question_res
        except Exception as e:
            logger.error(e)
            return reading_comprehension_question_res


def case_analysis_question_generation(data_helper, qa_gen):
    case_analysis_question_res = {"status": 0, "result": []}
    if qa_gen.caseAnalysisQuestion.questionNum == 0:
        logger.info("case analysis question num is 0")
        return None
    else:
        try:
            prompt = data_helper.prompt
            for model_name in ['gpt-4o', 'qwen-max', 'ERNIE-4.0-8K']:
                case_analysis_question_prompt = prompt['qa_generation'][
                    'case_analysis_question_generation']
                case_analysis_question_messages = data_helper.get_messages(
                    task_name="case_analysis_question_generation",
                    model_name=model_name,
                    system_prompt=case_analysis_question_prompt,
                    qa_gen=qa_gen)

                case_analysis_question_res["result"] = \
                    json.loads(llm.get_response(model_name=model_name, messages=case_analysis_question_messages))[
                        "result"]
                if qa_gen.caseAnalysisQuestion.questionNum * 4 != len(case_analysis_question_res['result']):
                    # 对案例分析的题目数量进行判断
                    logger.error(
                        f"The case analysis question generated is {case_analysis_question_res['result']}"
                        f" != parameters caseAnalysisQuestion.questionNum {qa_gen.caseAnalysisQuestion.questionNum}")
                    case_analysis_question_res["status"] = 0
                    return {"status": 0, "result": []}
                else:
                    logger.info(
                        f"case analysis  question result were generated successfully, The number of generated is {qa_gen.caseAnalysisQuestion.questionNum}, model name is {model_name}")
                    case_analysis_question_res["status"] = 1
                    return case_analysis_question_res
        except Exception as e:
            logger.error(e)
            return case_analysis_question_res


def qa_type_merging(futures):
    """
    :param futures: 每个题型作为单独的 futures返回 作为入参
    :return: 返回题目生成结果
    """
    result = {
        "choiceQuestion": {
            "status": 0,
            "result": []
        },
        "fillInTheBlankQuestion": {
            "status": 0,
            "result": []
        },
        "shortAnswerQuestion": {
            "status": 0,
            "result": []
        },
        "readingComprehensionQuestion": {
            "status": 0,
            "result": []
        },
        "caseAnalysisQuestion": {
            "status": 0,
            "result": []
        }
    }
    for future_item in as_completed(futures):
        future_res = future_item.result()
        if future_res is not None and future_res["status"] == 1:
            if type(future_res["result"][0]) is dict:
                if future_res["result"][0]['type'] == "填空题":
                    result["fillInTheBlankQuestion"] = future_res
                elif future_res["result"][0]['type'] == "简答题":
                    result["shortAnswerQuestion"] = future_res
                else:
                    result["choiceQuestion"] = future_res
            elif type(future_res["result"][0]) is list:
                result["readingComprehensionQuestion"] = future_res
            elif type(future_res["result"][0]) is str:
                result["caseAnalysisQuestion"] = future_res
            else:
                logger.error(f"futures result error ,res item is {future_res}")
                continue
        elif future_res is None:
            continue
        else:
            logger.warning(f"futures result error ,res item is {future_res}")
            continue
    return result


def send_result_to_frontend(callback_url: str, result: dict):
    try:
        response = requests.post(callback_url, json=result, headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            logger.info(f"{result['id']} Callback sent successfully!")
        else:
            logger.error(f"{result['id']} Callback failed with status code {response.status_code}")
    except Exception as e:
        logger.error(f"{result['id']} Error sending callback: {e}")


def qa_generation(qa_gen: QaGeneration):
    data_helper = DataHelper()
    futures = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures.append(executor.submit(choice_question_generation, data_helper, qa_gen))
        futures.append(executor.submit(fib_question_generation, data_helper, qa_gen))
        futures.append(executor.submit(short_answer_question_generation, data_helper, qa_gen))
        futures.append(executor.submit(reading_comprehension_question_generation, data_helper, qa_gen))
        futures.append(executor.submit(case_analysis_question_generation, data_helper, qa_gen))

    result = qa_type_merging(futures)
    result["id"], result["retryFlag"] = qa_gen.id, qa_gen.retryFlag
    return result


def decompose_knowledge_point(qa_gen):
    data_helper = DataHelper()
    prompt = data_helper.prompt
    try:
        for model_name in ['gpt-4o', 'qwen-max', 'ERNIE-4.0-8K']:
            system_prompt = prompt['decompose_knowledge']
            messages = data_helper.get_messages(task_name="decompose_knowledge_point", model_name=model_name,
                                                system_prompt=system_prompt, qa_gen=qa_gen)
            decompose_res = json.loads(llm.get_response(model_name=model_name, messages=messages))
            logger.info(f"decompose result is {decompose_res}")
            return decompose_res
    except Exception as e:
        logger.error(f"decompose knowledge point error, error is {e}, use original knowledge point")
        return {"result": [
            {'knowledgeTitle': qa_gen.knowledgeTitle, 'knowledgePoint': qa_gen.knowledgePoint, 'questionNum': 2}]}


def tag_generation(qa_gen):
    try:
        data_helper = DataHelper()
        prompt = data_helper.prompt
        for model_name in ['gpt-4o', 'qwen-max', 'ERNIE-4.0-8K']:
            system_prompt = prompt['qa_generation']['tag_generation']
            tag_messages = data_helper.get_messages(task_name="tag_generation", model_name=model_name,
                                                    system_prompt=system_prompt, qa_gen=qa_gen)
            llm_res = json.loads(llm.get_response(model_name=model_name, messages=tag_messages))["result"]
            logger.info(f"tag generation return success, model is {model_name}, result is {llm_res}")
            return llm_res
    except Exception as e:
        logger.error(e)
        return []


def process_qa_generation(qa_gen: QaGeneration):
    """
    正常流程的题目生成结果返回
    :param qa_gen: 接口入参
    :return: 结果回调前端接口
    """
    callback_url = "http://127.0.0.1:8080/jeecg-boot/course/question/generateQuestionsCallBack"
    try:
        result = qa_generation(qa_gen)
        # 需要设置前端的回调地址
        send_result_to_frontend(callback_url, result)

    except Exception as e:
        logger.error(f"{qa_gen.id} callback url{callback_url} error : {e}")


def convenient_qa_generation(qa_gen: QaGeneration, split_kg: list):
    result = {"id": qa_gen.id, "tagList": qa_gen.tagList, "retryFlag": qa_gen.retryFlag, "qa_result": []}
    logger.info("convenient qa generation start ...")
    try:
        for qa_item in split_kg:
            result_item = {
                qa_item["knowledgeTitle"]: {"splitKnowledgePoint": None, "questionNum": None, "result": None,
                                            "status": 0}}
            result_item[qa_item["knowledgeTitle"]]["splitKnowledgePoint"] = qa_item["knowledgePoint"]
            result_item[qa_item["knowledgeTitle"]]["questionNum"] = qa_item["questionNum"]
            if qa_gen.choiceQuestion.questionNum != 0:
                qa_gen.choiceQuestion.questionNum = qa_item["questionNum"]
            if qa_gen.shortAnswerQuestion.questionNum != 0:
                qa_gen.shortAnswerQuestion.questionNum = qa_item["questionNum"]
            if qa_gen.fillInTheBlankQuestion.questionNum != 0:
                qa_gen.fillInTheBlankQuestion.questionNum = qa_item["questionNum"]
            if qa_gen.readingComprehensionQuestion.questionNum != 0:
                qa_gen.readingComprehensionQuestion.questionNum = 1
            if qa_gen.caseAnalysisQuestion.questionNum != 0:
                qa_gen.caseAnalysisQuestion.questionNum = 1
            result_item[qa_item["knowledgeTitle"]]["result"] = qa_generation(qa_gen)
            result_item[qa_item["knowledgeTitle"]]["status"] = 1
            result["qa_result"].append(result_item)
    except Exception as e:
        logger.error(e)
    return result


def process_convenient_qa_generation(qa_gen: QaGeneration):
    """
    一键题目生成的结果返回
    :param qa_gen: 接口入参
    :return: 结果回调前端接口
    """
    callback_url = "http://127.0.0.1:8080/jeecg-boot/course/question/generateQuestionsCallBack"
    try:
        kg_tag_list = tag_generation(qa_gen)
        qa_gen.tagList = kg_tag_list
        split_kg = decompose_knowledge_point(qa_gen)

        result = convenient_qa_generation(qa_gen, split_kg["result"])
        send_result_to_frontend(callback_url, result)

    except Exception as e:
        logger.error(f"{qa_gen.id} callback url{callback_url} error : {e}")

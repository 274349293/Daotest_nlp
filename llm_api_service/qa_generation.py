from model.llm_service import LLMService
from pydantic import BaseModel, Field
from utils.nlp_logging import CustomLogger
import json
from typing import List, Optional

logger = CustomLogger(name="HuiRen qa generation api", write_to_file=True)
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
    knowledgeTitle: str
    knowledgePoint: str
    tagList: List[str] = Field(..., alias="tagList")  # 处理小写字段名
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

        if task_name == "choice_question_generation":
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
        else:
            # ReadingComprehensionQuestion and CaseAnalysisQuestion
            pass

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


def short_answer_question_generation(data_helper, qa_gen):
    short_answer_question_res = {"status": 0, "result": []}
    if qa_gen.shortAnswerQuestion.questionNum == 0:
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


def fib_question_generation(data_helper, qa_gen):
    fib_question_res = {"status": 0, "result": []}
    if qa_gen.fillInTheBlankQuestion.questionNum == 0:
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


def qa_merging(choice_question_res, fib_question_res, short_answer_question_res):
    result = {}
    if choice_question_res is not None:
        result["choiceQuestion"] = choice_question_res
    if fib_question_res is not None:
        result["fibQuestion"] = fib_question_res
    if short_answer_question_res is not None:
        result["shortAnswerQuestion"] = short_answer_question_res

    return result


def qa_generation(qa_gen: QaGeneration):
    data_helper = DataHelper()

    choice_question_res = choice_question_generation(data_helper, qa_gen)

    fib_question_res = fib_question_generation(data_helper, qa_gen)

    short_answer_question_res = short_answer_question_generation(data_helper, qa_gen)

    result = qa_merging(choice_question_res, fib_question_res, short_answer_question_res)
    return result

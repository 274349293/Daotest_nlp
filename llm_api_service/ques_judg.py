import ast
from pydantic import BaseModel
from model.llm_service import LLMService
# import llm_service
from utils.nlp_logging import CustomLogger

"""
汇仁智能培训app 多选题，填空题判断接口
tips：
1.多选题最多4个选项，填空题只有一个空位
2.先走filter ，filter 失败再走大模型做判断

logic：
1.选择过滤规则：给定list，做字典过滤
2.填空题每个空位的答案不多于5个字，判断大小写和阿拉伯数字与汉字数字-->字典映射

update:
1.不再走大模型了，只走逻辑规则
"""
logger = CustomLogger(name="HuiRen ques judgment api")
system_prompt = """汇仁是一家大型医药企业集团，汇仁公司新招聘了一批新员工，公司已经对新员工完成了入职培训，现在以考试的方式检查新员工是否已经掌握了培训内容。你是一名资深的培训经理，你的任务如下：{
1.判断员工作答的对错。考试题目全部为填空题，题目中的##代表所需要填写的空位，每个题目中只有1个空位。示例如下：{
题目：在与客户洽谈时间的安排上，要保证时间满足##要求。
标准答案：客户}，
2.判断对错的标准如下：
{
2.1 逐个比较员工作答每个空位的答案和标准答案中每个空位的答案，如果完全相同，则该空位标记为正确。如果不完全相同，但将答案填入题目中所表达的语义相同或95%以上相同，则该空位也标记为正确。
2.2 逐个比较员工作答每个空位的答案和标准答案中每个空位的答案，如果不完全相同，并且将答案填入题目中所表达的语义也不相同，则该空位标记为错误。
2.3 如果员工的作答和标准答案差距非常大，则该空位标记为错误。
2.4 用户作答中出现符号与标准答案不同的情况，如大小写不同，或者%写成“百分之”类似情况，只要语义和标准答案相同，一律判断为正确。
2.5 正确的填空为标记为1，错误的填空位标记为0。示例如下：{
题目：在与客户洽谈时间的安排上，要保证时间满足##要求。
标准答案：客户
用户作答：客户的
判断结果：1
}
}
3.回答的输出格式为固定格式，json格式，示例如下：
{
    "llm_result": 0
}
"""
llm = LLMService(llm_logger=logger)
# app = FastAPI()
choice_question_map = {'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D', 'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd'}

fb_question_map = {
    # 小写字母转大写
    'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D', 'e': 'E', 'f': 'F', 'g': 'G', 'h': 'H', 'i': 'I', 'j': 'J',
    'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'o': 'O', 'p': 'P', 'q': 'Q', 'r': 'R', 's': 'S', 't': 'T',
    'u': 'U', 'v': 'V', 'w': 'W', 'x': 'X', 'y': 'Y', 'z': 'Z',
    # 大写字母转小写
    'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd', 'E': 'e', 'F': 'f', 'G': 'g', 'H': 'h', 'I': 'i', 'J': 'j',
    'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'O': 'o', 'P': 'p', 'Q': 'q', 'R': 'r', 'S': 's', 'T': 't',
    'U': 'u', 'V': 'v', 'W': 'w', 'X': 'x', 'Y': 'y', 'Z': 'z',
    # 中文数字（小写）转阿拉伯数字
    '一': '1', '二': '2', '三': '3', '四': '4', '五': '5', '六': '6', '七': '7', '八': '8', '九': '9', '十': '10',
    '十一': '11', '十二': '12', '十三': '13', '十四': '14', '十五': '15', '十六': '16', '十七': '17', '十八': '18',
    '十九': '19',
    '二十': '20', '二十一': '21', '二十二': '22', '二十三': '23', '二十四': '24', '二十五': '25', '二十六': '26',
    '二十七': '27', '二十八': '28', '二十九': '29',
    '三十': '30', '三十一': '31', '三十二': '32', '三十三': '33', '三十四': '34', '三十五': '35', '三十六': '36',
    '三十七': '37', '三十八': '38', '三十九': '39',
    '四十': '40', '四十一': '41', '四十二': '42', '四十三': '43', '四十四': '44', '四十五': '45', '四十六': '46',
    '四十七': '47', '四十八': '48', '四十九': '49',
    '五十': '50', '五十一': '51', '五十二': '52', '五十三': '53', '五十四': '54', '五十五': '55', '五十六': '56',
    '五十七': '57', '五十八': '58', '五十九': '59',
    '六十': '60', '六十一': '61', '六十二': '62', '六十三': '63', '六十四': '64', '六十五': '65', '六十六': '66',
    '六十七': '67', '六十八': '68', '六十九': '69',
    '七十': '70', '七十一': '71', '七十二': '72', '七十三': '73', '七十四': '74', '七十五': '75', '七十六': '76',
    '七十七': '77', '七十八': '78', '七十九': '79',
    '八十': '80', '八十一': '81', '八十二': '82', '八十三': '83', '八十四': '84', '八十五': '85', '八十六': '86',
    '八十七': '87', '八十八': '88', '八十九': '89',
    '九十': '90', '九十一': '91', '九十二': '92', '九十三': '93', '九十四': '94', '九十五': '95', '九十六': '96',
    '九十七': '97', '九十八': '98', '九十九': '99',
    '一百': '100',
    # 中文数字（大写）转阿拉伯数字
    '壹': '1', '贰': '2', '叁': '3', '肆': '4', '伍': '5', '陆': '6', '柒': '7', '捌': '8', '玖': '9', '拾': '10',
    '拾壹': '11', '拾贰': '12', '拾叁': '13', '拾肆': '14', '拾伍': '15', '拾陆': '16', '拾柒': '17', '拾捌': '18',
    '拾玖': '19',
    '贰拾': '20', '贰拾壹': '21', '贰拾贰': '22', '贰拾叁': '23', '贰拾肆': '24', '贰拾伍': '25', '贰拾陆': '26',
    '贰拾柒': '27', '贰拾捌': '28', '贰拾玖': '29',
    '叁拾': '30', '叁拾壹': '31', '叁拾贰': '32', '叁拾叁': '33', '叁拾肆': '34', '叁拾伍': '35', '叁拾陆': '36',
    '叁拾柒': '37', '叁拾捌': '38', '叁拾玖': '39',
    '肆拾': '40', '肆拾壹': '41', '肆拾贰': '42', '肆拾叁': '43', '肆拾肆': '44', '肆拾伍': '45', '肆拾陆': '46',
    '肆拾柒': '47', '肆拾捌': '48', '肆拾玖': '49',
    '伍拾': '50', '伍拾壹': '51', '伍拾贰': '52', '伍拾叁': '53', '伍拾肆': '54', '伍拾伍': '55', '伍拾陆': '56',
    '伍拾柒': '57', '伍拾捌': '58', '伍拾玖': '59',
    '陆拾': '60', '陆拾壹': '61', '陆拾贰': '62', '陆拾叁': '63', '陆拾肆': '64', '陆拾伍': '65', '陆拾陆': '66',
    '陆拾柒': '67', '陆拾捌': '68', '陆拾玖': '69',
    '柒拾': '70', '柒拾壹': '71', '柒拾贰': '72', '柒拾叁': '73', '柒拾肆': '74', '柒拾伍': '75', '柒拾陆': '76',
    '柒拾柒': '77', '柒拾捌': '78', '柒拾玖': '79',
    '捌拾': '80', '捌拾壹': '81', '捌拾贰': '82', '捌拾叁': '83', '捌拾肆': '84', '捌拾伍': '85', '捌拾陆': '86',
    '捌拾柒': '87', '捌拾捌': '88', '捌拾玖': '89',
    '玖拾': '90', '玖拾壹': '91', '玖拾贰': '92', '玖拾叁': '93', '玖拾肆': '94', '玖拾伍': '95', '玖拾陆': '96',
    '玖拾柒': '97', '玖拾捌': '98', '玖拾玖': '99',
    '壹佰': '100',
    # other
    '1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九', '10': '十',
    '11': '十一', '12': '十二', '13': '十三', '14': '十四', '15': '十五', '16': '十六', '17': '十七', '18': '十八',
    '19': '十九',
    '20': '二十', '21': '二十一', '22': '二十二', '23': '二十三', '24': '二十四', '25': '二十五', '26': '二十六',
    '27': '二十七', '28': '二十八', '29': '二十九',
    '30': '三十', '31': '三十一', '32': '三十二', '33': '三十三', '34': '三十四', '35': '三十五', '36': '三十六',
    '37': '三十七', '38': '三十八', '39': '三十九',
    '40': '四十', '41': '四十一', '42': '四十二', '43': '四十三', '44': '四十四', '45': '四十五', '46': '四十六',
    '47': '四十七', '48': '四十八', '49': '四十九',
    '50': '五十', '51': '五十一', '52': '五十二', '53': '五十三', '54': '五十四', '55': '五十五', '56': '五十六',
    '57': '五十七', '58': '五十八', '59': '五十九',
    '60': '六十', '61': '六十一', '62': '六十二', '63': '六十三', '64': '六十四', '65': '六十五', '66': '六十六',
    '67': '六十七', '68': '六十八', '69': '六十九',
    '70': '七十', '71': '七十一', '72': '七十二', '73': '七十三', '74': '七十四', '75': '七十五', '76': '七十六',
    '77': '七十七', '78': '七十八', '79': '七十九',
    '80': '八十', '81': '八十一', '82': '八十二', '83': '八十三', '84': '八十四', '85': '八十五', '86': '八十六',
    '87': '八十七', '88': '八十八', '89': '八十九',
    '90': '九十', '91': '九十一', '92': '九十二', '93': '九十三', '94': '九十四', '95': '九十五', '96': '九十六',
    '97': '九十七', '98': '九十八', '99': '九十九',
    '100': '一百'

}


class QjInfo(BaseModel):
    """
    type 代表选择题和填空题的枚举类型
    { 选择题：0. 填空题：1 }
    """
    id: str
    question: str
    answer: str
    standardAnswer: str
    type: int


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
    return 0


def choice_question_judgment(options: str, standardAnswer: str):
    try:
        if len(options) > 4 or len(options) < 1:
            logger.error(f"choice question options len > 4 or <1, options: {options}")
            return 0
        for option in options:
            if option not in choice_question_map:
                logger.error(f"choice question options not in map dict, options: {options}")
                return 0
        if len(standardAnswer) == len(options):
            for standard_answer in standardAnswer:
                if standard_answer not in options:
                    logger.info(
                        f"The multiple choice option is incorrect:{options} , standard answer is {standard_answer}")
                    return 0
            logger.info(f"Multiple choice answer is correct")
            return 1
        else:
            logger.info(f"Multiple choice answer is incorrect, len(answer) != len(standardAnswer)")
            return 0
    except Exception as e:
        logger.error(f"choice_question_judgment error:  {e}")
        return 0


def fill_in_the_blanks(answer: str, standardAnswer: str, question: str, question_id: str):
    try:
        if standardAnswer != answer:
            if len(standardAnswer) != len(answer):
                if answer in fb_question_map and fb_question_map[answer] == standardAnswer:
                    logger.info(
                        f"fb judgment result is correct ,standardAnswer is {standardAnswer}, answer is {answer}")
                    return 1
                else:
                    # for model_name in ['gpt-4o', 'gpt-4o-mini', 'qwen-max', 'ERNIE-4.0-8K']:
                    #     try:
                    #         rag_content = {"题目": question, "标准答案": standardAnswer,
                    #                        "员工作答": answer}
                    #
                    #         if model_name != 'ERNIE-4.0-8K':
                    #             messages = [{'role': 'system', 'content': system_prompt},
                    #                         {'role': 'user',
                    #                          'content': f"以下是员工作答的题目和答案信息：{str(rag_content)}"}, ]
                    #         else:
                    #             messages = [
                    #                 {'role': 'user',
                    #                  'content': system_prompt + f"以下是员工作答的题目和答案信息：{str(rag_content)}"}]
                    #         completion = llm.get_response(model_name=model_name, messages=messages)
                    #
                    #         if model_name == 'gpt-4o' or model_name == 'gpt-4o-mini':
                    #             result = int(json.loads(completion)["llm_result"])
                    #             logger.info(f"fib return {question_id} success, model name is {model_name}")
                    #             logger.info(f"result is  {result}")
                    #             return result
                    #
                    #         elif model_name == 'qwen-max' or model_name == 'ERNIE-4.0-8K':
                    #             result = int(json_formatting_repair(completion)["llm_result"])
                    #             logger.info(f"fib return {question_id} success, model name is {model_name}")
                    #             logger.info(f"result is  {result}")
                    #             return result
                    #
                    #     except Exception as e:
                    #         logger.error(f"fib error in {model_name}: {e}")
                    # logger.error(f"{question_id} fib api return failed , return 0")
                    logger.info(
                        f"fb judgment result is incorrect, standardAnswer is {standardAnswer}, answer is {answer}")
                    return 0
            else:
                if answer in fb_question_map and fb_question_map[answer] == standardAnswer:
                    logger.info(
                        f"fb judgment result is correct ,standardAnswer is {standardAnswer}, answer is {answer}")
                    return 1
                for i in range(len(standardAnswer)):
                    if standardAnswer[i] == answer[i]:
                        continue
                    else:
                        if standardAnswer[i] in fb_question_map and answer[i] in fb_question_map:
                            if fb_question_map[answer[i]] == standardAnswer[i]:
                                continue
                            else:
                                logger.info(
                                    f"fb judgment result is incorrect, standardAnswer is {standardAnswer}, answer is {answer}")
                                return 0
                        else:
                            logger.info(
                                f"fb judgment result is incorrect, standardAnswer is {standardAnswer}, answer is {answer}")
                            return 0
                logger.info(f"fb judgment result is correct ,standardAnswer is {standardAnswer}, answer is {answer}")
                return 1
        else:
            logger.info(f"fb judgment result is correct, standardAnswer is {standardAnswer}, answer is {answer}")
            return 1
    except Exception as e:
        logger.error(f"fill_in_the_blanks error: {e}")
        return 0


def ques_judgment(qj_info: QjInfo):
    """
    :param qj_info:
    :return: 0 or 1 (0：incorrect，1：correct)
    """
    logger.info("------------------start--------------------")

    if qj_info.type == 0:
        logger.info(f"question type is choice")
        result = choice_question_judgment(qj_info.answer, qj_info.standardAnswer)
        return result

    elif qj_info.type == 1:
        logger.info(f"question type is fb")
        result = fill_in_the_blanks(qj_info.answer, qj_info.standardAnswer, qj_info.question, qj_info.id)
        return result

    else:
        logger.error(f"param type error: type is {qj_info.type} , return 0")
        return 0

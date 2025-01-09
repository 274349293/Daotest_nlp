import json
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from model.llm_service import LLMService
from utils.nlp_logging import CustomLogger

logger = CustomLogger("HuiRen practice stream api")
system_prompt = """汇仁是一家大型医药企业集团，汇仁公司新招聘了一批新员工，公司已经对新员工完成了入职培训，现在以考试的方式检查新员工是否已经掌握了培训内容。你是一名资深的培训经理，你的任务如下：{
1.对员工的作答评分，评分规定：满分为100分，最低分为0分，请从0-100直接给出一个整数作为员工作答分数。
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
5.在回复的前10个字当中不要出现 '评价' 这两个字
6.对知识点进行打分采取尽量宽容的形式，如果一个员工对一个知识点的作答可扣分也可不扣分，则不进行扣分，给满分。
7.在评价过程中不要说出具体哪个点获得多少分，不能让用户知道自己的作答的具体得分情况，只对用户作答的内容进行评价
8.回答的输出格式为固定格式，先输出员工作答的分数（一个0-100的整数），然后固定输出 ### ，最后输出评价部分。示例如下：

员工作答的分数###对员工作答的评价

"""
llm = LLMService(llm_logger=logger)


class PracticeQaInfo(BaseModel):
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


async def get_stream_response(qa_info: PracticeQaInfo):
    logger.info("--------------------start--------------------------")

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

    def wenxin_generate() -> bytes:
        for chunk in completion:
            try:
                if chunk['body']['is_end'] is True:
                    yield "data: [END]\n\n"
                else:
                    chunk_content = f"data: {chunk['body']['result']}\n\n"
                    yield chunk_content
            except Exception as e:
                yield "data: [END]\n\n"
                logger.error(f"yield [END], wenxin chunk is error, {e}")

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
            completion = llm.get_response_stream(model_name=model_name, messages=messages)

            if completion is None:
                continue
            logger.info(f"{qa_info.id} model name is {model_name} , get stream completion success")

            if model_name == 'gpt-4o':
                return StreamingResponse(chat_gpt_4o_generate(), media_type="text/event-stream")

            elif model_name == 'qwen-max':
                return StreamingResponse(qwen_generate(), media_type="text/event-stream")

            elif model_name == 'ERNIE-4.0-8K':
                return StreamingResponse(wenxin_generate(), media_type="text/event-stream")

        except Exception as e:
            logger.error(f"{qa_info.id} practice stream api error: {e}, model name is {model_name} ")
    logger.error(f"{qa_info.id} practice stream api return failed , return is None")
    return None

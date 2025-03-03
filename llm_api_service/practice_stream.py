import json
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from model.llm_service import LLMService
from utils.nlp_logging import CustomLogger


def get_prompt():
    with open('./utils/prompt.json', encoding='utf-8') as prompt:
        return json.load(prompt)


logger = CustomLogger("DaoTest practice stream api")
system_prompt = get_prompt()["practice_stream"]

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

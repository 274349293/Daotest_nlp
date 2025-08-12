from fastapi.responses import StreamingResponse
import json
import re
from model.llm_service import LLMService
from pydantic import BaseModel
from utils.nlp_logging import CustomLogger
from typing import List, Dict

"""
高尔夫小程序LLM接口
提供高尔夫相关问题的智能回复服务，支持流式返回

功能：
1. 接收用户的高尔夫相关问题
2. 基于对话历史提供专业的高尔夫知识回复
3. 支持多轮对话
4. 流式返回，提升用户体验

update0724 口语化回复优化，多模型api备用
update0730 重新拼接chunk,返回的chunk不再是按照接收到的返回，而是重新按句子拼接
"""

logger = CustomLogger(name="Golf LLM API", write_to_file=True)
llm = LLMService(llm_logger=logger)


class DialogueMessage(BaseModel):
    role: str  # system, assistant, user
    content: str


class GolfLLMInfo(BaseModel):
    id: str
    history: List[DialogueMessage]


def get_golf_system_prompt() -> str:
    """从配置文件获取高尔夫专业系统提示词"""
    try:
        with open('./utils/prompt.json', encoding='utf-8') as config_file:
            prompt_config = json.load(config_file)
            return prompt_config.get("golf_llm", "你是一位专业的高尔夫教练，请为用户提供专业的高尔夫指导。")
    except Exception as e:
        logger.error(f"Failed to load golf prompt config: {e}")
        return "你是一位专业的高尔夫教练，请为用户提供专业的高尔夫指导。"


def check_sentence_complete(text: str) -> tuple:
    """检查文本中是否有完整的句子，返回(完整句子列表, 剩余文本)"""
    # 定义句子结束符号
    sentence_endings = r'[。！？；，,.!?;]'

    # 找到所有句子结束位置
    matches = list(re.finditer(sentence_endings, text))

    if not matches:
        # 没有完整句子
        return [], text

    # 有完整句子
    complete_sentences = []
    last_end = 0

    for match in matches:
        sentence = text[last_end:match.end()].strip()
        if sentence:
            complete_sentences.append(sentence)
        last_end = match.end()

    # 剩余的不完整部分
    remaining = text[last_end:].strip()

    return complete_sentences, remaining


def process_golf_messages(golf_info: GolfLLMInfo) -> List[Dict[str, str]]:
    """处理高尔夫对话消息，确保格式正确并添加系统提示词"""
    processed_messages = []

    try:
        golf_system_prompt = get_golf_system_prompt()

        if (golf_info.history and
                golf_info.history[0].role == "system"):
            processed_messages.append({
                "role": "system",
                "content": golf_system_prompt
            })
            for msg in golf_info.history[1:]:
                processed_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        else:
            processed_messages.append({
                "role": "system",
                "content": golf_system_prompt
            })
            for msg in golf_info.history:
                processed_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        logger.info(f"Processed golf messages successfully for ID: {golf_info.id}")
        logger.info(f"Total messages: {len(processed_messages)}")

        return processed_messages

    except Exception as e:
        logger.error(f"Error processing golf messages: {e}")
        return []


def validate_golf_info(golf_info: GolfLLMInfo) -> bool:
    """验证高尔夫对话信息的完整性"""
    if not golf_info.id:
        logger.warning("Missing ID in golf info")
        return False

    if not golf_info.history:
        logger.warning("Empty history in golf info")
        return False

    user_messages = [msg for msg in golf_info.history if msg.role == "user"]
    if not user_messages:
        logger.warning("No user messages found in history")
        return False

    return True


async def golf_llm_chat(golf_info: GolfLLMInfo):
    """高尔夫LLM聊天主函数，支持流式返回，按句子重新拼接chunk"""

    def qwen_generate() -> bytes:
        buffer = ""
        full_response = ""
        end_sent = False  # 添加标志位防止重复发送[END]

        try:
            for chunk in completion:
                try:
                    content = json.loads(chunk.model_dump_json())['choices'][0]['delta']['content']
                    buffer += content
                    full_response += content

                    # 检查buffer中是否有完整的句子
                    complete_sentences, new_buffer = check_sentence_complete(buffer)

                    # 立即发送所有完整的句子
                    if complete_sentences:
                        for sentence in complete_sentences:
                            if sentence.strip():
                                chunk_content = f"data: {sentence}\n\n"
                                yield chunk_content

                        # 更新buffer为剩余内容
                        buffer = new_buffer

                except Exception as e:
                    logger.error(f"qwen chunk processing error: {e}")
                    break

            # 发送最后的buffer内容（可能是不完整的句子）
            if buffer.strip():
                yield f"data: {buffer}\n\n"

        except Exception as e:
            logger.error(f"qwen_generate error: {e}")
        finally:
            # 确保只发送一次[END]
            if not end_sent:
                logger.info(f"Complete model response: {full_response}")
                yield "data: [END]\n\n"
                end_sent = True

    def chat_gpt_4o_generate() -> bytes:
        buffer = ""
        full_response = ""
        end_sent = False  # 添加标志位防止重复发送[END]

        try:
            for chunk in completion:
                try:
                    if len(chunk.choices) == 0:
                        continue

                    if chunk.choices[0].delta.content is None:
                        # 流正常结束，发送剩余内容
                        if buffer.strip():
                            yield f"data: {buffer}\n\n"
                        break
                    else:
                        content = chunk.choices[0].delta.content
                        buffer += content
                        full_response += content

                        # 检查buffer中是否有完整的句子
                        complete_sentences, new_buffer = check_sentence_complete(buffer)

                        if complete_sentences:
                            # 立即发送所有完整的句子
                            for sentence in complete_sentences:
                                if sentence.strip():
                                    chunk_content = f"data: {sentence}\n\n"
                                    yield chunk_content

                            # 更新buffer为剩余内容
                            buffer = new_buffer

                except Exception as e:
                    logger.error(f"chat_gpt_4o chunk processing error: {e}")
                    break

        except Exception as e:
            logger.error(f"chat_gpt_4o_generate error: {e}")
        finally:
            # 确保只发送一次[END]
            if not end_sent:
                logger.info(f"Complete model response: {full_response}")
                yield "data: [END]\n\n"
                end_sent = True

    def wenxin_generate() -> bytes:
        buffer = ""
        full_response = ""
        end_sent = False  # 添加标志位防止重复发送[END]

        try:
            for chunk in completion:
                try:
                    if chunk['body']['is_end'] is True:
                        # 发送剩余内容并结束
                        if buffer.strip():
                            yield f"data: {buffer}\n\n"
                        break
                    else:
                        content = chunk['body']['result']
                        buffer += content
                        full_response += content

                        # 检查buffer中是否有完整的句子
                        complete_sentences, new_buffer = check_sentence_complete(buffer)

                        # 立即发送所有完整的句子
                        if complete_sentences:
                            for sentence in complete_sentences:
                                if sentence.strip():
                                    chunk_content = f"data: {sentence}\n\n"
                                    yield chunk_content

                            # 更新buffer为剩余内容
                            buffer = new_buffer

                except Exception as e:
                    logger.error(f"wenxin chunk processing error: {e}")
                    break

        except Exception as e:
            logger.error(f"wenxin_generate error: {e}")
        finally:
            # 确保只发送一次[END]
            if not end_sent:
                logger.info(f"Complete model response: {full_response}")
                yield "data: [END]\n\n"
                end_sent = True

    logger.info("------------------golf llm chat start--------------------")
    logger.info(f"Golf Chat ID: {golf_info.id}")

    if not validate_golf_info(golf_info):
        logger.error(f"Golf info validation failed for {golf_info.id}")
        return "error"

    processed_history = process_golf_messages(golf_info)

    if not processed_history:
        logger.error(f"Failed to process golf messages for {golf_info.id}")
        return "error"

    logger.info(f"Processed history length: {len(processed_history)}")

    for model_name in ['gpt-4o', 'gpt-4o-mini', 'qwen-max', 'deepseek-chat']:
        try:
            completion = llm.get_response_stream(model_name=model_name, messages=processed_history)
            if completion is None:
                logger.warning(f"{model_name} returned None, trying next model")
                continue

            logger.info(f"{golf_info.id} model name is {model_name}, get stream completion success")

            if model_name in ['gpt-4o', 'gpt-4o-mini', 'deepseek-chat']:
                return StreamingResponse(chat_gpt_4o_generate(), media_type="text/event-stream")
            elif model_name == 'qwen-max':
                return StreamingResponse(qwen_generate(), media_type="text/event-stream")
            elif model_name == 'ERNIE-4.0-8K':
                return StreamingResponse(wenxin_generate(), media_type="text/event-stream")

        except Exception as e:
            logger.error(f"golf llm chat error in {model_name}: {e}")
            continue

    logger.error(f"{golf_info.id} golf llm chat api return failed - all models exhausted, return is error")
    return "error"

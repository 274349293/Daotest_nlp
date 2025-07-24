from fastapi.responses import StreamingResponse
import json
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

update0724 口语化回复优化，多模型备用
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
        # 返回默认提示词作为备用
        return "你是一位专业的高尔夫教练，请为用户提供专业的高尔夫指导。"


def process_golf_messages(golf_info: GolfLLMInfo) -> List[Dict[str, str]]:
    """处理高尔夫对话消息，确保格式正确并添加系统提示词"""
    processed_messages = []

    try:
        # 获取高尔夫专业系统提示词
        golf_system_prompt = get_golf_system_prompt()

        # 检查第一条消息是否为system消息
        if (golf_info.history and
                golf_info.history[0].role == "system"):
            # 如果已有system消息，替换为高尔夫专业提示词
            processed_messages.append({
                "role": "system",
                "content": golf_system_prompt
            })
            # 添加其余消息
            for msg in golf_info.history[1:]:
                processed_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        else:
            # 如果没有system消息，先添加高尔夫专业提示词
            processed_messages.append({
                "role": "system",
                "content": golf_system_prompt
            })
            # 添加所有消息
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

    # 检查是否有用户消息
    user_messages = [msg for msg in golf_info.history if msg.role == "user"]
    if not user_messages:
        logger.warning("No user messages found in history")
        return False

    return True


async def golf_llm_chat(golf_info: GolfLLMInfo):
    """高尔夫LLM聊天主函数，支持流式返回"""

    def qwen_generate():
        """千问模型流式生成器"""
        for chunk in completion:
            try:
                chunk_content = f"data: {json.loads(chunk.model_dump_json())['choices'][0]['delta']['content']}\n\n"
                yield chunk_content
            except Exception as e:
                yield "data: [END]\n\n"
                logger.info(f"yield [END], qwen chunk is None, {e}")

    def chat_gpt_4o_generate():
        """GPT-4o模型流式生成器"""
        llm_resp = ""
        for chunk in completion:
            try:
                if len(chunk.choices) == 0:
                    continue
                if chunk.choices[0].delta.content is None:
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

    logger.info("------------------golf llm chat start--------------------")
    logger.info(f"Golf Chat ID: {golf_info.id}")

    # 验证对话信息的完整性
    if not validate_golf_info(golf_info):
        logger.error(f"Golf info validation failed for {golf_info.id}")
        return "error"

    # 处理对话历史
    processed_history = process_golf_messages(golf_info)

    if not processed_history:
        logger.error(f"Failed to process golf messages for {golf_info.id}")
        return "error"

    logger.info(f"Processed history length: {len(processed_history)}")

    # 按优先级尝试不同模型
    for model_name in ['gpt-4o', 'gpt-4o-mini', 'qwen-max']:
        try:
            completion = llm.get_response_stream(model_name=model_name, messages=processed_history)
            if completion is None:
                continue

            logger.info(f"{golf_info.id} model name is {model_name}, get stream completion success")

            if model_name in ['gpt-4o', 'gpt-4o-mini']:
                return StreamingResponse(chat_gpt_4o_generate(), media_type="text/event-stream")
            elif model_name == 'qwen-max':
                return StreamingResponse(qwen_generate(), media_type="text/event-stream")

        except Exception as e:
            logger.error(f"golf llm chat error in {model_name}: {e}")

    logger.error(f"{golf_info.id} golf llm chat api return failed, return is error")
    return "error"

import json
import ast
from model.llm_service import LLMService
from pydantic import BaseModel
from utils.nlp_logging import CustomLogger
from typing import List, Dict

"""
优化后的案例分析评价接口
这是multi_round_dialogue_mark的优化版本，参考llm_chat的优化思路

主要改进：
1. 简化入参结构，直接使用标准的对话历史格式
2. 支持动态的system prompt配置
3. 增强了数据验证和错误处理
4. 更清晰的代码结构和日志记录

update:
1. 参考llm_chat的优化方式，简化了参数处理逻辑
2. 新增case、question、answer字段支持
3. 使用prompt.json中的chat_rating配置进行系统提示词管理
4. 返回格式：{"score": "30", "llmResult": "评价内容"} 或 "error"
"""

logger = CustomLogger(name="DaoTest chat rating api", write_to_file=True)
llm = LLMService(llm_logger=logger)


class DialogueMessage(BaseModel):
    role: str  # system, assistant, user
    content: str


class ChatRatingInfo(BaseModel):
    id: str
    case: str  # 案例内容
    question: str  # 问题
    answer: str  # 标准答案
    history: List[DialogueMessage]  # 对话历史


def get_prompt_config():
    """获取prompt配置"""
    try:
        with open('./utils/prompt.json', encoding='utf-8') as config_file:
            return json.load(config_file)
    except Exception as e:
        logger.error(f"Failed to load prompt config: {e}")
        return {}


def format_dialogue_history(history: List[DialogueMessage]) -> str:
    """格式化对话历史为文本"""
    formatted_history = []
    for msg in history:
        if msg.role == "user":
            formatted_history.append(f"学员：{msg.content}")
        elif msg.role == "assistant":
            formatted_history.append(f"培训师：{msg.content}")
    return "\n".join(formatted_history)


def get_rating_system_prompt(case: str = "", question: str = "", answer: str = "", history: str = "") -> str:
    """获取评价系统提示词，并动态插入案例、问题、答案和对话历史"""
    prompt_config = get_prompt_config()

    # 从prompt.json中获取chat_rating的系统提示词
    system_prompt = prompt_config.get("chat_rating", "")

    if not system_prompt:
        # 如果配置文件中没有chat_rating，使用默认的评价提示词
        logger.warning("No chat_rating prompt found in config, using default")
        system_prompt = """你是一位经验丰富的培训评价专家，负责对学员的案例分析对话进行评分和评价。

案例内容：[case]
问题：[question]  
标准答案：[answer]
对话记录：[history]

请根据对话历史，对学员的整体表现进行评价。输出格式为JSON格式：
{
"score": "评分",
"llmResult": "评价内容"
}"""

    # 动态插入案例、问题、答案和对话历史
    if case:
        system_prompt = system_prompt.replace("[case]", case)
    if question:
        system_prompt = system_prompt.replace("[question]", question)
    if answer:
        system_prompt = system_prompt.replace("[answer]", answer)
    if history:
        system_prompt = system_prompt.replace("[history]", history)

    return system_prompt


def process_rating_messages(rating_info: ChatRatingInfo) -> List[Dict[str, str]]:
    """处理评价消息，构建LLM调用所需的消息格式"""
    try:
        # 格式化对话历史
        formatted_history = format_dialogue_history(rating_info.history)

        # 获取评价系统提示词
        rating_system_prompt = get_rating_system_prompt(
            rating_info.case,
            rating_info.question,
            rating_info.answer,
            formatted_history
        )

        # 构建消息列表
        messages = [
            {
                "role": "system",
                "content": rating_system_prompt
            },
            {
                "role": "user",
                "content": "请对上述对话中学员的表现进行评价。"
            }
        ]

        logger.info(f"Processed rating messages successfully for {rating_info.id}")
        logger.info(f"Case: {rating_info.case[:100]}...")  # 记录案例前100字符
        logger.info(f"Question: {rating_info.question}")
        logger.info(f"Dialogue messages count: {len(rating_info.history)}")

        return messages

    except Exception as e:
        logger.error(f"Error processing rating messages: {e}")
        return []


def validate_rating_info(rating_info: ChatRatingInfo) -> bool:
    """验证评价信息的完整性"""
    if not rating_info.case or not rating_info.question or not rating_info.answer:
        logger.warning(
            f"Missing required fields - case: {bool(rating_info.case)}, "
            f"question: {bool(rating_info.question)}, answer: {bool(rating_info.answer)}"
        )
        return False

    if not rating_info.history or len(rating_info.history) < 2:
        logger.warning(f"Insufficient dialogue history - length: {len(rating_info.history)}")
        return False

    # 检查是否有用户回答
    user_responses = [msg for msg in rating_info.history if msg.role == "user"]
    if not user_responses:
        logger.warning("No user responses found in dialogue history")
        return False

    return True


def json_formatting_repair(json_str: str):
    """
    修复JSON格式问题，针对qwen、wenxin生成的回复
    """
    try:
        json_str = "{" + ''.join(json_str.split('{')[1:])
        json_str = ''.join(json_str.split('}')[0]) + "}"
        json_str = ast.literal_eval(json_str)
        return json_str
    except Exception as e:
        logger.error(f"json_formatting_repair error: {e}")
        return None


def chat_rating(rating_info: ChatRatingInfo):
    """优化后的案例分析评价主函数"""
    logger.info("------------------chat rating start--------------------")
    logger.info(f"Rating ID: {rating_info.id}")

    # 验证评价信息的完整性
    if not validate_rating_info(rating_info):
        logger.error(f"Rating info validation failed for {rating_info.id}")
        return "error"

    # 处理评价消息
    messages = process_rating_messages(rating_info)

    if not messages:
        logger.error(f"Failed to process rating messages for {rating_info.id}")
        return "error"

    logger.info(f"Processed messages length: {len(messages)}")

    # 按优先级尝试不同模型
    for model_name in ['gpt-4o', 'gpt-4o-mini', 'qwen-max']:
        try:
            completion = llm.get_response(model_name=model_name, messages=messages)
            if completion is None:
                continue

            logger.info(f"{rating_info.id} model name is {model_name}, get completion success")

            if model_name in ['gpt-4o', 'gpt-4o-mini']:
                try:
                    result = json.loads(completion)
                    logger.info(f"Chat rating result: {result}")
                    return result
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error for {model_name}: {e}")
                    continue

            elif model_name == 'qwen-max':
                result = json_formatting_repair(completion)
                if result:
                    logger.info(f"Chat rating result: {result}")
                    return result
                else:
                    continue

        except Exception as e:
            logger.error(f"Chat rating error in {model_name}: {e}")

    logger.error(f"{rating_info.id} chat rating api return failed, return error")
    return "error"

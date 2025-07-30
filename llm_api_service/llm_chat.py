from fastapi.responses import StreamingResponse
import json
from model.llm_service import LLMService
from pydantic import BaseModel
from utils.nlp_logging import CustomLogger
from typing import List, Dict, Optional

"""
优化后的多轮对话接口
支持不同场景的对话：
0: 视频互动对话 (轮数定为2轮）
1: 案例分析对话 (轮数定为4轮)

主要改进：
1. 支持多种场景，每种场景有固定的system prompt
2. 入参结构优化，更清晰明确
3. 保持原有的流式返回特性
4. 新增case字段支持案例分析场景

update:
1. 新增case字段，用于案例分析对话场景
2. 增强了prompt模板的动态插入功能
3. 0730 取消markdown形式
"""

logger = CustomLogger(name="DaoTest llm chat api", write_to_file=True)
llm = LLMService(llm_logger=logger)


class DialogueMessage(BaseModel):
    role: str  # system, assistant, user
    content: str


class OptimizedDialogueInfo(BaseModel):
    id: str
    scene: int  # 0: 视频互动对话, 1: 案例分析对话
    question: str = ""  # 问题内容
    answer: str = ""  # 参考答案
    case: str = ""  # 案例内容（新增字段，主要用于场景1）
    history: List[DialogueMessage]


def get_prompt_config():
    """获取prompt配置"""
    try:
        with open('./utils/prompt.json', encoding='utf-8') as config_file:
            return json.load(config_file)
    except Exception as e:
        logger.error(f"Failed to load prompt config: {e}")
        return {}


def get_scene_system_prompt(scene: int, question: str = "", answer: str = "", case: str = "") -> str:
    """根据场景获取对应的system prompt，并动态插入问题、答案和案例"""
    prompt_config = get_prompt_config()
    scene_prompts = prompt_config.get("llm_chat", {})

    scene_key = str(scene)
    if scene_key in scene_prompts:
        system_prompt = scene_prompts[scene_key]

        # 对于场景0（视频互动对话），需要插入问题和答案
        if scene == 0 and question and answer:
            system_prompt = system_prompt.replace("[question]", question)
            system_prompt = system_prompt.replace("[answer]", answer)

        # 对于场景1（案例分析对话），需要插入案例、问题和答案
        elif scene == 1:
            if case:
                system_prompt = system_prompt.replace("[case]", case)
            if question:
                system_prompt = system_prompt.replace("[question]", question)
            if answer:
                system_prompt = system_prompt.replace("[answer]", answer)

        return system_prompt
    else:
        logger.warning(f"Scene {scene} not found in config, using default")
        # 返回默认的system prompt
        return scene_prompts.get("default", "你是一位专业的培训助手，请根据对话历史继续回复。")


def process_dialogue_history(dialogue_info: OptimizedDialogueInfo) -> List[Dict[str, str]]:
    """处理对话历史，确保格式正确"""
    processed_history = []

    try:
        # 获取场景对应的system prompt，并插入问题、答案和案例
        scene_system_prompt = get_scene_system_prompt(
            dialogue_info.scene,
            dialogue_info.question,
            dialogue_info.answer,
            dialogue_info.case  # 新增案例参数
        )

        # 检查第一条消息是否为system消息
        if (dialogue_info.history and
                dialogue_info.history[0].role == "system"):
            # 如果已有system消息，替换为场景对应的system prompt
            processed_history.append({
                "role": "system",
                "content": scene_system_prompt
            })
            # 添加其余消息
            for msg in dialogue_info.history[1:]:
                processed_history.append({
                    "role": msg.role,
                    "content": msg.content
                })
        else:
            # 如果没有system消息，先添加场景对应的system prompt
            processed_history.append({
                "role": "system",
                "content": scene_system_prompt
            })
            # 添加所有消息
            for msg in dialogue_info.history:
                processed_history.append({
                    "role": msg.role,
                    "content": msg.content
                })

        logger.info(f"Processed dialogue history successfully for scene {dialogue_info.scene}")

        # log test
        if dialogue_info.scene == 1:
            logger.info(f"Case analysis dialogue - Case: {dialogue_info.case[:100]}...")
            logger.info(f"Case analysis dialogue - Question: {dialogue_info.question}")
            logger.info(f"Case analysis dialogue - Answer: {dialogue_info.answer[:100]}...")

        return processed_history

    except Exception as e:
        logger.error(f"Error processing dialogue history: {e}")
        return []


def validate_dialogue_info(dialogue_info: OptimizedDialogueInfo) -> bool:
    """验证对话信息的完整性"""
    if dialogue_info.scene == 0:
        # 视频互动对话需要question和answer
        if not dialogue_info.question or not dialogue_info.answer:
            logger.warning(
                f"Scene 0 missing required fields - question: {bool(dialogue_info.question)}, answer: {bool(dialogue_info.answer)}")
            return False
    elif dialogue_info.scene == 1:
        # 案例分析对话需要case、question和answer
        if not dialogue_info.case or not dialogue_info.question or not dialogue_info.answer:
            logger.warning(
                f"Scene 1 missing required fields - case: {bool(dialogue_info.case)}, question: {bool(dialogue_info.question)}, answer: {bool(dialogue_info.answer)}")
            return False

    return True


async def optimized_multi_round_dialogue(dialogue_info: OptimizedDialogueInfo):
    """优化后的多轮对话主函数"""

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

    logger.info("------------------llm chat start--------------------")
    logger.info(f"Dialogue ID: {dialogue_info.id}, Scene: {dialogue_info.scene}")

    # 验证对话信息的完整性
    if not validate_dialogue_info(dialogue_info):
        logger.error(f"Dialogue info validation failed for {dialogue_info.id}")
        return "error"

    # 处理对话历史
    processed_history = process_dialogue_history(dialogue_info)

    if not processed_history:
        logger.error(f"Failed to process dialogue history for {dialogue_info.id}")
        return "error"

    logger.info(f"Processed history length: {len(processed_history)}")

    # 按优先级尝试不同模型
    for model_name in ['gpt-4o', 'gpt-4o-mini', 'qwen-max']:
        try:
            completion = llm.get_response_stream(model_name=model_name, messages=processed_history)
            if completion is None:
                continue

            logger.info(f"{dialogue_info.id} model name is {model_name}, get stream completion success")

            if model_name in ['gpt-4o', 'gpt-4o-mini']:
                return StreamingResponse(chat_gpt_4o_generate(), media_type="text/event-stream")
            elif model_name == 'qwen-max':
                return StreamingResponse(qwen_generate(), media_type="text/event-stream")

        except Exception as e:
            logger.error(f"optimized mr dialogue error in {model_name}: {e}")

    logger.error(f"{dialogue_info.id} optimized mr dialogue api return failed, return is error")
    return "error"

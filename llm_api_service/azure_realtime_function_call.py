import json
import re
import math
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from utils.nlp_logging import CustomLogger
from collections import Counter

logger = CustomLogger(name="DaoTest realtime function call api", write_to_file=True)


class FunctionCallQuery(BaseModel):
    user_input: str  # 用户输入的内容
    function_call_name: str  # 函数调用名称


class RealtimeFunctionCallInfo(BaseModel):
    topic: str  # 主题/课程名
    action: str  # 动作: startSession 或 getFunctionCallResult
    query: Optional[FunctionCallQuery] = None  # 函数调用查询信息，仅在action=getFunctionCallResult时需要


class RealtimeFunctionCallService:
    def __init__(self):
        self.prompt_config = self.load_prompt_config()
        self.courses_config = self.prompt_config.get("azure_realtime_function_call", {})

    @staticmethod
    def load_prompt_config():
        """加载prompt配置文件"""
        try:
            with open('./utils/prompt.json', encoding='utf-8') as config_file:
                return json.load(config_file)
        except Exception as e:
            logger.error(f"Failed to load prompt config file: {e}")
            return {}

    def get_course_config(self, topic: str) -> Optional[Dict[str, Any]]:
        """根据topic获取对应的课程配置"""
        # 必须完全匹配
        if topic in self.courses_config:
            return self.courses_config[topic]

        return None

    def start_session(self, topic: str) -> Dict[str, Any]:
        """启动会话，返回指令和工具定义"""
        course_config = self.get_course_config(topic)

        if course_config:
            return {
                "instructions": course_config.get("instructions", ""),
                "tools": course_config.get("tools", [])
            }
        else:
            # 如果没有找到对应的课程配置，返回空字典
            return {}

    def hybrid_retrieve(self, query: str, course_config: Dict[str, Any]) -> Dict[str, Any]:
        """混合检索策略 - 参考原始代码的实现"""
        structured_knowledge = course_config.get("structured_knowledge", {})
        full_content = course_config.get("full_training_content", "")

        if not structured_knowledge:
            return {
                "matched": False,
                "message": "No structured knowledge available.",
                "full_content": full_content
            }

        # 确保query是字符串
        if not isinstance(query, str):
            query = str(query)

        query_lower = query.lower()

        # 第一步：尝试关键词映射（参考原始代码）
        topic_mapping = {
            "新出海": "新出海概述",
            "出海概述": "新出海概述",
            "出海趋势": "新出海概述",
            "趋势": "新出海概述",

            "出海路径": "出海四条路径",
            "出海方式": "出海四条路径",
            "四条路径": "出海四条路径",
            "路径": "出海四条路径",

            "要不要出海": "要不要出海",
            "是否出海": "要不要出海",
            "出海决策": "要不要出海",
            "适合出海": "要不要出海",

            "何时出海": "何时出海",
            "出海时机": "何时出海",
            "什么时候出海": "何时出海",
            "时机": "何时出海",

            "如何出海": "如何出海",
            "出海方法": "如何出海",
            "方法": "如何出海",

            "出海条件": "出海必要条件",
            "必要条件": "出海必要条件",
            "出海要求": "出海必要条件",
            "条件": "出海必要条件",

            "文化融合": "文化融合是通行证",
            "文化适应": "文化融合是通行证",
            "通行证": "文化融合是通行证",
            "融合": "文化融合是通行证",

            "商务谈判": "商务谈判是护身符",
            "法律谈判": "商务谈判是护身符",
            "护身符": "商务谈判是护身符",
            "谈判": "商务谈判是护身符",

            "五新": "五新概览",
            "新基建": "五新概览",
            "新能源": "五新概览",
            "新智造": "五新概览",
            "新消费": "五新概览"
        }

        # 尝试直接匹配
        for key in topic_mapping:
            if key in query_lower:
                matched_topic = topic_mapping[key]
                if matched_topic in structured_knowledge:
                    logger.info(f"Keyword mapping found match: {matched_topic}")
                    return {
                        "matched": True,
                        "topic": matched_topic,
                        "content": structured_knowledge[matched_topic].strip()
                    }

        # 第二步：如果关键词映射失败，使用综合评分
        query_words = re.findall(r'\w+', query_lower)
        scores = {}

        for topic_name, content in structured_knowledge.items():
            score = 0
            content_lower = content.lower()
            topic_name_lower = topic_name.lower()

            # 1. 标题完全匹配 (权重: 10)
            if topic_name_lower == query_lower:
                score += 10

            # 2. 标题包含查询或查询包含标题 (权重: 5)
            elif query_lower in topic_name_lower or topic_name_lower in query_lower:
                score += 5

            # 3. 标题中的词出现在查询中 (权重: 3)
            topic_words = re.findall(r'\w+', topic_name_lower)
            for word in topic_words:
                if len(word) > 1 and word in query_lower:
                    score += 3 / len(topic_words)  # 按词数平均

            # 4. 内容关键词匹配 (权重: 2)
            matched_words = sum(1 for word in query_words if len(word) > 1 and word in content_lower)
            if query_words and len(query_words) > 0:
                coverage = matched_words / len(query_words)
                score += coverage * 2

            # 5. 特殊关键词加分
            if any(keyword in query_lower for keyword in ["出海", "海外", "国际", "全球化", "走出去"]):
                if any(keyword in content_lower for keyword in ["出海", "海外", "国际", "全球", "跨境"]):
                    score += 0.5

            scores[topic_name] = score

        # 找出得分最高的主题
        if scores:
            best_topic = max(scores.items(), key=lambda x: x[1])
            best_topic_name, best_score = best_topic

            # 降低阈值，提高匹配成功率
            threshold = 0.5  # 从1.5降低到0.5
            if best_score >= threshold:
                logger.info(f"Hybrid retrieval found match: {best_topic_name} with score {best_score:.2f}")
                return {
                    "matched": True,
                    "topic": best_topic_name,
                    "content": structured_knowledge[best_topic_name].strip()
                }

        # 第三步：如果还是没有匹配，检查是否包含出海相关关键词
        if any(keyword in query_lower for keyword in ["出海", "海外", "国际", "全球化", "走出去"]):
            # 返回新出海概述作为默认
            if "新出海概述" in structured_knowledge:
                logger.info(f"Default match for going global keywords")
                return {
                    "matched": True,
                    "topic": "新出海概述",
                    "content": structured_knowledge["新出海概述"].strip()
                }

        logger.info(f"No match found for query: {query}")
        return {
            "matched": False,
            "message": "未找到与问题直接相关的主题，提供完整培训内容作为参考。",
            "full_content": full_content
        }

    def execute_function(self, user_input: str, func_name: str, course_config: Dict[str, Any]) -> Dict[str, Any]:
        """执行指定的函数"""
        # 验证函数名是否匹配课程配置中的function_call_name
        expected_func_name = course_config.get("function_call_name", "")
        if func_name != expected_func_name:
            logger.warning(f"Function name mismatch: expected {expected_func_name}, got {func_name}")

        # 使用混合检索策略
        return self.hybrid_retrieve(user_input, course_config)

    def get_function_call_result(self, query: FunctionCallQuery, topic: str) -> Dict[str, Any]:
        """获取函数调用结果"""
        # 获取课程配置
        course_config = self.get_course_config(topic)
        if not course_config:
            return {"error": f"Course config not found for topic '{topic}'"}

        func_name = query.function_call_name
        user_input = query.user_input

        logger.info(f"Processing function call: {func_name} with user input: {user_input}")

        # 执行函数并获取结果
        func_result = self.execute_function(user_input, func_name, course_config)

        # 返回格式为 {函数名: 结果}
        return {func_name: func_result}


def realtime_function_call(fc_info: RealtimeFunctionCallInfo) -> Dict[str, Any]:
    """处理实时函数调用接口"""
    logger.info("------------------start--------------------")
    logger.info(f"Received realtime function call request: topic={fc_info.topic}, action={fc_info.action}")

    try:
        service = RealtimeFunctionCallService()

        if fc_info.action == "startSession":
            # 启动会话
            result = service.start_session(fc_info.topic)
            logger.info(f"Start session success")
            return result

        elif fc_info.action == "getFunctionCallResult":
            # 获取函数调用结果
            if not fc_info.query:
                error_msg = "Must provide query parameter when getting function call result"
                logger.error(error_msg)
                return {"error": error_msg}

            result = service.get_function_call_result(fc_info.query, fc_info.topic)
            logger.info(f"Get function call result success")
            return result

        else:
            logger.error(f"Action error, undefined action: {fc_info.action}")
            return {}

    except Exception as e:
        error_msg = f"Error occurred while processing realtime function call: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}

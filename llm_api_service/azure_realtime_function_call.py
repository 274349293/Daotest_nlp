import json
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from utils.nlp_logging import CustomLogger

logger = CustomLogger(name="DaoTest realtime function call api", write_to_file=True)


class RealtimeFunctionCallInfo(BaseModel):
    topic: str  # 主题/课程名
    action: str  # 动作: startSession 或 getFunctionCallResult
    name: Optional[List[str]] = None  # 触发的function call的名字列表，仅在action=getFunctionCallResult时需要


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
            logger.error(f" Course {topic} not exist")
            # 如果没有找到对应的课程配置，返回默认响应
            return {}

    def execute_function(self, func_name: str, topic: str, course_config: Dict[str, Any]) -> Dict[str, Any]:
        """执行指定的函数"""
        # 根据不同的函数名执行不同的检索逻辑
        if func_name == "get_enterprise_going_global_info":
            return self.retrieve_knowledge(topic, course_config)
        elif func_name == "get_sales_skills_info":
            return self.retrieve_knowledge(topic, course_config)
        elif func_name == "get_project_management_info":
            return self.retrieve_knowledge(topic, course_config)
        else:
            return {"error": f"Unknown function: {func_name}"}

    def retrieve_knowledge(self, query: str, course_config: Dict[str, Any]) -> Dict[str, Any]:
        """检索知识库"""
        structured_knowledge = course_config.get("structured_knowledge", {})
        full_content = course_config.get("full_training_content", "")

        # 简单的关键词匹配检索
        matched_topic = None
        for topic_name, content in structured_knowledge.items():
            if topic_name.lower() in query.lower() or query.lower() in topic_name.lower():
                matched_topic = topic_name
                break
            # 检查内容中是否包含查询关键词
            if any(keyword in content for keyword in query.split() if len(keyword) > 2):
                matched_topic = topic_name
                break

        if matched_topic:
            return {
                "matched": True,
                "topic": matched_topic,
                "content": structured_knowledge[matched_topic].strip(),
                "method": "Keyword matching retrieval"
            }
        else:
            return {
                "matched": False,
                "message": "No directly related topic found, providing complete training content as reference.",
                "full_content": full_content,
                "method": "Keyword matching retrieval"
            }

    def get_function_call_result(self, name_list: List[str], topic: str) -> Dict[str, Any]:
        """获取函数调用结果"""
        result = {}

        # 获取课程配置
        course_config = self.get_course_config(topic)
        if not course_config:
            return {"error": f"Course config not found for topic '{topic}'"}

        # 执行每个函数
        for func_name in name_list:
            result[func_name] = self.execute_function(func_name, topic, course_config)

        return result


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
            if not fc_info.name:
                error_msg = "Must provide name parameter when getting function call result"
                logger.error(error_msg)
                return {"error": error_msg}

            result = service.get_function_call_result(fc_info.name, fc_info.topic)
            logger.info(f"Get function call result success")
            return result

        else:
            logger.error(f"Action error, undefined action: {fc_info.action}")
            return {}

    except Exception as e:
        error_msg = f"Error occurred while processing realtime function call: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}

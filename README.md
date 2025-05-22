# Daotest_nlp

汇仁智能培训项目 - 自然语言处理服务

## 项目简介

Daotest_nlp 是一个基于多个大语言模型（GPT-4o、通义千问、文心一言）的智能培训系统，提供题目判断、行为风格测试、考试评分、练习评价、多轮对话、标签生成、题目生成等多种NLP功能。

## 主要功能

### 核心服务接口

1. **题目判断服务** (`/ques_judgment`)
   - 支持选择题和填空题的自动判断
   - 基于规则引擎和映射字典进行快速判分

2. **行为风格测试** (`/behavioral_style_test`)
   - 基于18题问卷的行为风格分析
   - 自动计算并返回用户行为风格类型（社交型、关系型、指导型、思考型）

3. **考试评分服务** (`/exam_mark`)
   - 智能评分简答题和阅读理解题
   - 基于标准答案进行多维度评价

4. **练习评价服务** (`/practice_stream`)
   - 流式返回练习评价结果
   - 支持实时反馈和指导

5. **多轮对话服务** (`/mr_dialogue`)
   - 案例分析题的多轮对话交互
   - 引导式学习，最多4轮对话

6. **对话评价服务** (`/mr_dialogue_mark`)
   - 对多轮对话内容进行评分和评价

7. **标签生成服务** (`/tag_generation`)
   - 基于知识内容自动生成特征标签
   - 支持自定义标签数量

8. **题目生成服务** (`/qa_generation`)
   - 支持多种题型：选择题、填空题、简答题、阅读理解、案例分析
   - 异步处理，支持批量生成

9. **知识点拆分服务** (`/decompose_knowledge`)
   - 将大段知识点拆分为独立的知识模块

10. **实时语音服务** (`/realtime_function_call`)
    - Azure实时语音模型的function call结果返回
    - 支持企业出海、销售技巧、项目管理等多个培训主题

## 技术架构

### 后端框架
- **FastAPI**: 现代、快速的Web框架
- **Python 3.x**: 主要开发语言

### 大语言模型集成
- **GPT-4o/GPT-4o-mini**: Azure OpenAI服务
- **通义千问 (Qwen-max)**: 阿里云大模型服务
- **文心一言 (ERNIE-4.0-8K)**: 百度千帆大模型服务

### 核心依赖
```
fastapi==0.68.0
openai==1.3.0
qianfan==0.2.6
uvicorn==0.15.0
jieba==0.42.1
```

### 智能检索优化
- **稀疏检索**: 基于BM25算法的关键词匹配
- **密集检索**: 基于Sentence Transformers的语义相似度计算
- **查询理解**: 意图分类、实体识别、查询扩展
- **缓存机制**: 模型缓存和结果缓存优化性能

## 安装与运行

### 环境要求
- Python 3.8+
- 依赖包见 `requirement.txt`

### 安装步骤

1. 克隆项目
```bash
git clone https://github.com/ToClannadQAQ/Daotest_nlp.git
cd Daotest_nlp
```

2. 安装依赖
```bash
pip install -r requirement.txt
```

3. 配置模型API密钥
编辑 `utils/model_config.json`，填入相应的API密钥和端点信息：

```json
{
  "chatgpt-4o": {
    "api_key": "your_azure_openai_key",
    "azure_endpoint": "your_azure_endpoint",
    "api_version": "2024-08-01-preview"
  },
  "qwen": {
    "api_key": "your_qwen_api_key",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
  },
  "wenxin": {
    "api_key": "your_wenxin_api_key",
    "secret_key": "your_wenxin_secret_key"
  }
}
```

4. 启动服务
```bash
python api_services.py
```

服务将在 `http://0.0.0.0:9000` 启动

## API 接口说明

### 题目判断接口
```http
POST /ques_judgment
Content-Type: application/json

{
    "id": "question_id",
    "question": "题目内容",
    "answer": "用户答案",
    "standardAnswer": "标准答案",
    "type": 0  // 0:选择题, 1:填空题
}
```

### 行为风格测试接口
```http
POST /behavioral_style_test
Content-Type: application/json

{
    "A1": 1, "B1": 2,
    "A2": 1, "B2": 2,
    // ... 18对题目的答案
}
```

### 题目生成接口
```http
POST /qa_generation
Content-Type: application/json

{
    "id": "task_id",
    "type": 1,  // 0:便捷生成, 1:标准生成
    "knowledgeTitle": "知识点标题",
    "knowledgePoint": "知识点内容",
    "tagList": ["标签1", "标签2"],
    "choiceQuestion": {
        "questionNum": 5,
        "additionalPrompt": "额外要求"
    }
    // ... 其他题型配置
}
```

### 实时语音接口
```http
POST /realtime_function_call
Content-Type: application/json

{
    "topic": "企业出海",  // 支持: 企业出海、销售技巧、项目管理
    "action": "getFunctionCallResult",
    "query": {
        "user_input": "用户问题",
        "function_call_name": "get_enterprise_going_global_info"
    }
}
```

## 项目结构

```
Daotest_nlp/
├── api_services.py              # FastAPI主服务文件
├── llm_api_service/            # LLM服务模块
│   ├── ques_judg.py            # 题目判断服务
│   ├── behavioral_style_test.py # 行为风格测试
│   ├── exam_marking.py         # 考试评分服务
│   ├── practice_stream.py      # 练习评价服务
│   ├── multi_round_dialogue.py # 多轮对话服务
│   ├── qa_generation.py        # 题目生成服务
│   ├── tag_generation.py       # 标签生成服务
│   ├── decompose_knowledge_point.py # 知识点拆分
│   └── azure_realtime_function_call.py # 实时语音服务
├── model/
│   └── llm_service.py          # LLM统一服务封装
├── utils/
│   ├── model_config.json       # 模型配置文件
│   ├── prompt.json            # 提示词配置
│   ├── nlp_logging.py         # 日志模块
│   └── log/                   # 日志目录
├── requirement.txt            # 项目依赖
└── README.md                 # 项目说明文档
```

## 配置说明

### 模型配置 (`utils/model_config.json`)
包含各个大语言模型的API配置信息，支持多模型冗余和故障转移。

### 提示词配置 (`utils/prompt.json`)
包含各种任务的系统提示词模板，支持灵活的提示词管理和优化。

### 实时语音配置
支持多个培训主题的结构化知识库：
- **企业出海**: 出海策略、路径、条件等
- **销售技巧**: SPIN销售法、异议处理、成交技巧等  
- **项目管理**: 五大过程组、风险管理、敏捷方法等

## 性能优化

### 检索优化
- **模型缓存**: 全局缓存避免重复加载
- **快速配置**: 优先使用稀疏检索，按需启用密集检索
- **内容限制**: 限制处理内容长度提升响应速度
- **异步处理**: 支持异步检索和并发处理

### 服务优化  
- **模型冗余**: 多模型支持，自动故障转移
- **流式响应**: 支持流式输出，提升用户体验
- **异步任务**: 耗时任务异步处理，支持回调通知

## 日志系统

集成完整的日志记录系统：
- 按日期轮转的日志文件
- 多级别日志输出（INFO、DEBUG、WARNING、ERROR）
- 控制台和文件双重输出
- 详细的API调用和错误追踪

## 开发指南

### 添加新的LLM服务
1. 在 `llm_api_service/` 目录下创建新的服务文件
2. 继承 `LLMService` 类进行模型调用
3. 在 `api_services.py` 中注册新的API端点
4. 更新 `utils/prompt.json` 添加相应的提示词

### 扩展培训主题
1. 在 `utils/prompt.json` 的 `azure_realtime_function_call` 中添加新主题
2. 配置结构化知识库和主题映射
3. 更新函数调用工具定义

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 贡献

欢迎提交 Issue 和 Pull Request 来改进项目。

## 联系方式

- 项目维护者: ToClannadQAQ
- GitHub: https://github.com/ToClannadQAQ/Daotest_nlp

---

**注意**: 使用前请确保已正确配置所有必要的API密钥和服务端点。
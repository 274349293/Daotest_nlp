from fastapi import FastAPI
from openai import OpenAI
from fastapi.responses import StreamingResponse

app = FastAPI()

# 初始化你的OpenAI客户端，确保有正确的API密钥和配置
client = OpenAI(
    api_key="sk-gl5n4T1iBsCcDGhl7r61T9dZdK1JLg4h3Tjb0WINONfVGSGR",
    base_url="https://api.moonshot.cn/v1",
)


@app.post("/chat_stream")
async def chat_stream():
    # 构造请求大模型的消息内容
    messages = [
        {
            "role": "system",
            "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。",
        },
        {"role": "user", "content": "你好，我叫李雷，1+1等于多少？"},
    ]

    # 发起带有流式响应的大模型请求
    completion = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=messages,
        temperature=0.3,
        stream=True
    )

    async def generate():
        """
        异步生成器函数，用于从大模型的流式响应中逐条读取并发送内容。
        注意：此处逻辑依赖于OpenAI SDK的具体实现细节，特别是如何正确处理stream=True时的响应。
        """
        for chunk in completion:  # 假定completion能直接async for循环，实际根据SDK文档调整
            # 检查chunk结构并提取所需部分，这取决于API的实际响应格式
            delta_content = chunk.choices[0].delta.content  # 获取内容，如果有的话
            print(delta_content, type(delta_content))
            if type(delta_content) is str:
                yield delta_content  # 将文本内容转换为字节并yield出去

    # 使用StreamingResponse返回流式数据
    return StreamingResponse(
        content=generate(),
        media_type="text/event-stream",  # 适合于实时数据流的MIME类型
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

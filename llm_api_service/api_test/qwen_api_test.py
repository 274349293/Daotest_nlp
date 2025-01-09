import json

from openai import OpenAI
import os


def get_response():
    client = OpenAI(
        api_key="sk-824ae94382af45a9857c20853793dcbb",  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务的base_url
    )
    completion = client.chat.completions.create(
        model="qwen-max",
        messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
                  {'role': 'user', 'content': '你是谁？'}]
    )
    print(json.loads(completion.model_dump_json()))


def get_stream_response():
    client = OpenAI(
        api_key="sk-824ae94382af45a9857c20853793dcbb",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-max",
        messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
                  {'role': 'user', 'content': '你是谁？'}],
        stream=True,
        # 可选，配置以后会在流式输出的最后一行展示token使用信息
        stream_options={"include_usage": True}
    )
    for chunk in completion:
        try:
            chunk_content = json.loads(chunk.model_dump_json())['choices'][0]['delta']['content']
            print(chunk_content)
        except Exception as e:
            pass


if __name__ == '__main__':
    get_stream_response()

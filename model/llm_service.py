import json
from openai import OpenAI
from openai import AzureOpenAI
import time
from utils.nlp_logging import CustomLogger
import os
import qianfan


class LLMService:
    def __init__(self, llm_logger=CustomLogger()):
        self.logger = llm_logger
        self.config = self.get_config()
        self.model_dict = {"gpt-4o", "qwen-max", "ERNIE-4.0-8K", "gpt-4o-mini", "deepseek-chat"}

    @staticmethod
    def get_config():
        with open('./utils/model_config.json') as config_file:
            return json.load(config_file)

    def qwen_response(self, model_name="qwen-max", message=None):
        client = OpenAI(
            api_key=self.config['qwen']['api_key'],
            base_url=self.config['qwen']['base_url'],
        )
        try:
            completion = client.chat.completions.create(
                temperature=0.05,
                model=model_name,
                messages=message,
                stream=False,
                response_format={"type": "json_object"}
            )
            return json.loads(completion.model_dump_json())['choices'][0]['message']['content']
        except Exception as e:
            self.logger.error(f"qwen response error: {e}")
            return None

    def qwen_response_stream(self, model_name="qwen-max", message=None):
        client = OpenAI(
            api_key=self.config['qwen']['api_key'],
            base_url=self.config['qwen']['base_url'],
        )
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=message,
                stream=True,
                stream_options={"include_usage": True}
            )
            return completion
        except Exception as e:
            self.logger.error(f"qwen stream response error: {e}")
            return None

    def deepseek_response(self, model_name="deepseek-chat", message=None):
        """DeepSeek API 调用方法 - 非流式"""
        client = OpenAI(
            api_key=self.config['deepseek']['api_key'],
            base_url=self.config['deepseek']['base_url'],
        )
        try:
            completion = client.chat.completions.create(
                temperature=0.05,
                model=model_name,
                messages=message,
                stream=False,
                response_format={"type": "json_object"}
            )
            return json.loads(completion.model_dump_json())['choices'][0]['message']['content']
        except Exception as e:
            self.logger.error(f"deepseek response error: {e}")
            return None

    def deepseek_response_stream(self, model_name="deepseek-chat", message=None):
        """DeepSeek API 调用方法 - 流式"""
        client = OpenAI(
            api_key=self.config['deepseek']['api_key'],
            base_url=self.config['deepseek']['base_url'],
        )
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=message,
                stream=True,
                stream_options={"include_usage": True}
            )
            return completion
        except Exception as e:
            self.logger.error(f"deepseek stream response error: {e}")
            return None

    def chatgpt_4o_response(self, model_name="gpt-4o", message=None):
        client = AzureOpenAI(
            api_key=self.config['chatgpt-4o']['api_key'],
            azure_endpoint=self.config['chatgpt-4o']['azure_endpoint'],
            api_version=self.config['chatgpt-4o']['api_version'],
        )
        input_message = message

        def chat_completion(retry_count=3, llm_message=input_message):
            if retry_count == 0:
                return None
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    temperature=0.05,
                    max_tokens=4096,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    messages=llm_message,
                    response_format={"type": "json_object"},
                    stream=False)

                res_content = response.choices[0].message.content.strip().rstrip("<|im_end|>")

                return res_content
            except Exception as e:
                self.logger.error(f"chatgpt 4o response error: {e}")
                time.sleep(1)
                return chat_completion(retry_count - 1)

        try:
            completion = chat_completion(retry_count=3)
            return completion
        except Exception as e:
            self.logger.error(f"chatgpt_4o_response error: {e}")
            return None

    def chatgpt_4o_response_stream(self, model_name="gpt-4o", message=None):
        client = AzureOpenAI(
            api_key=self.config['chatgpt-4o']['api_key'],
            azure_endpoint=self.config['chatgpt-4o']['azure_endpoint'],
            api_version=self.config['chatgpt-4o']['api_version'],
        )
        input_message = message

        def chat_completion(retry_count=3, llm_message=input_message):
            if retry_count == 0:
                return None
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    temperature=0.05,
                    max_tokens=4096,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    messages=llm_message,
                    stream=True)

                return response
            except Exception as e:
                self.logger.error(f"chatgpt 4o response_stream error: {e}, retry num is {retry_count}")
                time.sleep(1)
                return chat_completion(retry_count - 1)

        try:
            completion = chat_completion(retry_count=3)
            return completion
        except Exception as e:
            self.logger.error(f"chatgpt_4o_response_stream error: {e}")
            return None

    def chatgpt_4o_mini_response(self, model_name="gpt-4o-mini", message=None):
        client = AzureOpenAI(
            api_key=self.config['chatgpt-4o-mini']['api_key'],
            azure_endpoint=self.config['chatgpt-4o-mini']['azure_endpoint'],
            api_version=self.config['chatgpt-4o-mini']['api_version'],
        )
        input_message = message

        def chat_completion(retry_count=3, llm_message=input_message):
            if retry_count == 0:
                return None
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    temperature=0.05,
                    max_tokens=4096,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    messages=llm_message,
                    response_format={"type": "json_object"},
                    stream=False)

                res_content = response.choices[0].message.content.strip().rstrip("<|im_end|>")

                return res_content
            except Exception as e:
                self.logger.error(f"chatgpt 4o mini response error: {e}")
                time.sleep(1)
                return chat_completion(retry_count - 1)

        try:
            completion = chat_completion(retry_count=3)
            return completion
        except Exception as e:
            self.logger.error(f"chatgpt 4o mini response error: {e}")
            return None

    def chatgpt_4o_mini_response_stream(self, model_name="gpt-4o-mini", message=None):
        client = AzureOpenAI(
            api_key=self.config['chatgpt-4o-mini']['api_key'],
            azure_endpoint=self.config['chatgpt-4o-mini']['azure_endpoint'],
            api_version=self.config['chatgpt-4o-mini']['api_version'],
        )
        input_message = message

        def chat_completion(retry_count=3, llm_message=input_message):
            if retry_count == 0:
                return None
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    temperature=0.05,
                    max_tokens=4096,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    messages=llm_message,
                    stream=True)

                return response
            except Exception as e:
                self.logger.error(f"chatgpt 4o mini response_stream error: {e}, retry num is {retry_count}")
                time.sleep(1)
                return chat_completion(retry_count - 1)

        try:
            completion = chat_completion(retry_count=3)
            return completion
        except Exception as e:
            self.logger.error(f"chatgpt 4o mini response_stream error: {e}")
            return None

    def wenxin_response(self, model_name="ERNIE-4.0-8K", message=None):

        os.environ["QIANFAN_AK"] = self.config['wenxin']['api_key']
        os.environ["QIANFAN_SK"] = self.config['wenxin']['secret_key']
        try:
            client = qianfan.ChatCompletion()
            completion = client.do(model=model_name, messages=message)

            return completion["body"]['result']
        except Exception as e:
            self.logger.error(f"wenxin response error: {e}")
            return None

    def wenxin_response_stream(self, model_name="ERNIE-4.0-8K", message=None):

        os.environ["QIANFAN_AK"] = self.config['wenxin']['api_key']
        os.environ["QIANFAN_SK"] = self.config['wenxin']['secret_key']
        try:
            client = qianfan.ChatCompletion()
            completion = client.do(model=model_name, messages=message, stream=True)

            return completion
        except Exception as e:
            self.logger.error(f"wenxin response stream error: {e}")
            return None

    def get_response(self, model_name=None, messages=None):
        """
        :param model_name: {"gpt-4o","qwen-max","deepseek-chat"}
        :param stream: Ture or False
        :return: response str
        """

        if messages is not None and type(messages) == list and len(messages) > 0:
            self.logger.info(f"messages: {messages}")
            message = messages
        else:
            self.logger.warning(f"input messages error,messages is {messages}")
        if model_name not in self.model_dict:
            self.logger.warning(f"{model_name} not in {self.model_dict}, default gpt-4o")
            return self.chatgpt_4o_response(model_name=model_name, message=message)
        else:
            if model_name == "gpt-4o":
                return self.chatgpt_4o_response(model_name=model_name, message=message)
            elif model_name == "gpt-4o-mini":
                return self.chatgpt_4o_mini_response(model_name=model_name, message=message)
            elif model_name == "qwen-max":
                return self.qwen_response(model_name=model_name, message=message)
            elif model_name in ["deepseek-chat"]:
                return self.deepseek_response(model_name=model_name, message=message)
            elif model_name == "ERNIE-4.0-8K":
                return self.wenxin_response(model_name=model_name, message=message)
            else:
                self.logger.warning(f"model error , {model_name} not in {self.model_dict}, default gpt-4o")
                return self.chatgpt_4o_response(model_name=model_name, message=message)

    def get_response_stream(self, model_name=None, messages=None):
        """

        :param model_name:
        :param messages:
        :return: llm stream response str
        """

        if messages is not None and type(messages) == list and len(messages) > 0:
            self.logger.info(f"messages: {messages}")
            message = messages
        else:
            self.logger.warning(f"input messages error,messages is {messages}")
        if model_name not in self.model_dict:
            self.logger.warning(f"{model_name} not in {self.model_dict}, default gpt-4o")
            return self.chatgpt_4o_response(model_name=model_name, message=message)
        else:
            if model_name == "gpt-4o":
                return self.chatgpt_4o_response_stream(model_name=model_name, message=message)
            elif model_name == "gpt-4o-mini":
                return self.chatgpt_4o_mini_response_stream(model_name=model_name, message=message)
            elif model_name == "qwen-max":
                return self.qwen_response_stream(model_name=model_name, message=message)
            elif model_name in ["deepseek-chat"]:
                return self.deepseek_response_stream(model_name=model_name, message=message)
            elif model_name == "ERNIE-4.0-8K":
                return self.wenxin_response_stream(model_name=model_name, message=message)
            else:
                self.logger.warning(f"model error , {model_name} not in {self.model_dict}, default gpt-4o")
                return self.chatgpt_4o_response(model_name=model_name, message=message)


if __name__ == '__main__':
    pass

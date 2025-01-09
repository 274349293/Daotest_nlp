import os
import qianfan

# 【推荐】使用安全认证AK/SK鉴权，通过环境变量初始化认证信息
# 替换下列示例中参数，安全认证Access Key替换your_iam_ak，Secret Key替换your_iam_sk
os.environ["QIANFAN_AK"] = "qGmPd8UlxI6xfGeTry7Sw5bL"
os.environ["QIANFAN_SK"] = "j6n6MzEuGQSFG5OgIjnOtWGxPGaPn9Yh"

chat_comp = qianfan.ChatCompletion()

# 指定特定模型
resp = chat_comp.do(model="ERNIE-3.5-8K", messages=[{
    "role": "user",
    "content": "你好"
}])

print(resp["body"]['result'])

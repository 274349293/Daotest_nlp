import openai
import time
import os

from openai import AzureOpenAI

OPENAI_KEY = "" if os.environ.get('OPENAI_KEY') is None else os.environ.get('OPENAI_KEY')

# OpenAI服务
client = AzureOpenAI(
    api_key="5fea49cd1d9b404598ed9d2259738486",
    azure_endpoint="https://ll274349293.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?",
    api_version="2024-08-01-preview",
)


def chat_completion(messages, model="gpt-4o", temperature=0.85, retry_count=5):
    if retry_count == 0:
        return ""
    try:
        print(messages)
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=1200,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            messages=messages)
        # print(response)
        res_content = response.choices[0].message.content.strip().rstrip("<|im_end|>")
        print(res_content)
        return res_content
    except Exception as e:
        print(e)
        return chat_completion(messages, temperature, retry_count - 1)


def chat_completion_stream(messages, model="gpt-4o", temperature=0, max_tokens=4000):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,  # this is the degree of randomness of the model's output
        max_tokens=max_tokens,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=True
    )
    # create variables to collect the stream of chunks
    collected_chunks = []
    collected_messages = []
    # iterate through the stream of events
    for chunk in response:
        if (len(chunk.choices)) == 0:
            continue
        if chunk.choices[0].delta.content == None:
            continue
        collected_chunks.append(chunk)  # save the event response
        chunk_message = chunk.choices[0].delta.content  # extract the message
        collected_messages.append(chunk_message)  # save the message
        # partial_reply_content = ''.join([m.get('content', '') for m in collected_messages])
        print(chunk_message + "\n", end="", flush=True)
        # print(chunk_message)
    full_reply_content = ''.join([m for m in collected_messages])
    print(full_reply_content)
    return full_reply_content


def multi_round_dialogue_stream(messages, model="gpt-4o", temperature=0, max_tokens=4000):
    history = messages[:]  # 复制初始消息
    while True:
        user_input = input("用户: ")  # 接收用户输入
        if user_input.lower() in ['exit', 'quit']:  # 退出条件
            break
        history.append({'role': 'user', 'content': user_input})  # 添加用户消息

        # 调用模型获取回复
        response = chat_completion(messages=history, model=model, temperature=temperature)

        # 添加模型的回复到历史记录
        history.append({'role': 'assistant', 'content': response})

        print(f"助手: {response}\n")  # 输出助手的回复


if __name__ == '__main__':
    # prompt = """
    #     你作为汇仁的培训经理，任务如下：
    # {
    #     1.根据材料，出一道阅读理解的题目
    #     2.阅读理解分为材料，题目和标准答案3个部分
    #     3.题目和标准答案结合材料一起出，一共出3道或4道题目
    #     4.总结材料，输出的材料内容为原材料字数的50%左右
    #     5.题目为简答题，题目要有深度，标准答案的字数每道题超过150字，答案中标有1.2.3类似的知识点，答案中的知识点最少1个最多5个
    #     6.出题的方向和主题为： 思考，是为了更好地表达
    #
    # }
    # 材料如下：
    # {
    #     牛津大学里有一条被称作“哲学之路”的小路。这条小路是位于由日本皇太子曾经就读过的马顿校舍和牛津大学最有名的克莱斯特校舍之间的细小的石板路。
    #
    #     我当时选修的课程叫作“18世纪的英国哲学”。走进教室的时候，只有三四名学生坐在那里，其中还有一位老年人。过了一会儿，走进来一位很有哲学家风范的教授，并且马上就开始授课。不过惭愧的是，我完全听不懂教授在讲些什么。不管怎么集中注意力，因为授课内容包含了大量的专业词，我甚至觉得自己听到的不是英语。正因为我曾经在美国留过学，而且已经比较适应牛津大学的学习环境，所以更受打击。此后我还是坚持去听课，但是每次都只能茫然地坐在那里而已。
    #
    #     如今，在牛津大学，哲学仍然是一门备受尊崇的学问。
    #
    #     因为原本哲学就被誉为所有学问的基础。出身自牛津大学的伟大科学家们也都是哲学家。探究事物和人类本质的哲学，被人们定位为先于现代经济学、经营学、教育学、理工学等所有的学问。
    #
    #     在深奥的哲学课课堂上，笔者渐渐体会到了以下三点。
    #
    #     ①“拼搏奋斗”本身就有意义
    #
    #     即使听不懂课程内容，即使理解不了教科书的内容，也要把自己当作一张白纸，去拼命学习。这种行为本身是非常重要的。
    #
    #     我有一位毕业于牛津大学商学院的朋友。通过和他的交谈，我发现原来在牛津大学所学习的哲学在商业场合中也是非常有用的。
    #
    #     商业本身就是一种不断解决问题的持续拼搏。要想找到解决问题的线索，除了分析以往曾经发生过的类似案例以外，还可以分析同一个行业的案例，甚至完全不同行业的案例。通过这种全面的分析，有时候还会有全新的发现。
    #
    #     据说许多企业的管理者都有自己的哲学和历史观，他们能够从大局角度来提出敏锐的建议，并且往往都能奏效。在牛津大学主修历史和哲学的人也能活跃于商业领域，这或许就是其秘诀所在。
    #
    #
    #
    #     ②培养能够克服两难境地的思考能力
    #
    #     因“白热教室”而闻名全球的哈佛大学的迈克尔·桑德尔教授也毕业于牛津大学。他的课主要采用与学生“对话”的方式进行，非常具有活力，深受学生喜爱，教室里总是座无虚席。
    #
    #     桑德尔就读于牛津大学时，曾经研究过“政治哲学”。关于“哲学”这门深奥晦涩的学问，该如何让它变得简单易懂，让教学充满热情呢？桑德尔的政治哲学这门课的特征在于：以人类不得不面对的“两难境地”为题材，在与学生进行讨论的同时，让其加深对人类本质的理解。所谓的两难境地，简而言之就是左右为难、不知如何是好的情况。
    #
    #     让我们来看一看桑德尔的课程内容吧（参考：《关于今后的“正义”》，迈克尔·桑德尔著，鬼泽忍译，早川书房出版）。
    #
    #     对于以下的情况，大家会怎样思考呢？
    #
    #     【状况1】
    #
    #     印度有个贫穷的农夫，他的儿子很想去念大学，但是没有学费，他只好将自己其中一个肾脏切除，并卖给了急着给女儿进行脏器移植手术的美国人。这个美国人的女儿患有重病，但是移植肾脏后就有治愈的希望。
    #
    #     恐怕对于这样的行为，有不少人都会认为“能够让两个家庭都变得幸福，这也是无可奈何的吧”。但是，这个故事还没有结束。
    #
    #     【状况2】
    #
    #     其实，这个农夫还有一个儿子，也想要去念大学。此时，又有一个希望购买肾脏的人出现了，他希望用高价买下农夫剩下的一个肾脏。因为农夫只剩下一个肾脏了，一旦切除就会丧命。
    #
    #     请大家设身处地地想一想，你会允许【状况2】发生吗？还是持否定态度呢？
    #
    #     桑德尔的课程往往是将这样的两难境地放在现实环境中，让学生们去思考什么是“正义”，同时告诉学生们这样持续思考的重要性。
    #
    #     综上所述，牛津学子最重视的能力之一，就是不论面对多么绝望的状况，都要不停地进行思考，直至最后一刻。
    #
    #     ③培养发现问题的能力
    #
    #     哲学追求的是人类的“理想”，它的使命就在于思考实现这个目的的方法。在日常生活中，这样的思考训练最终能够培养起我们发现问题的能力。
    #
    #     这里所说的“问题”是指“理想”和“现实”之间的“差距”。只有认识到“差距”，才能发现问题。因为“理想”可以理解为“应有的状态”，所以如果对人类“应有的状态”不关心的话，就无法察觉到“问题”的所在。
    #
    #     如今，日本的大学非常重视培养能够在实际社会中马上运用的知识与技能。这样一来，学生就会往这些实用性强的学科上扎堆。
    #
    #     与之相对，在牛津大学里哲学之所以能够受众人尊敬，是因为每个师生都在思考“事物与人类的本质”，希望解答由此产生的“问题”，而这种问题是功利性的知识所不能解答的。
    # }
    #
    #     """
    # chat_completion(messages=[{'role': 'system',
    #                            'content': '汇仁是一家大型医药企业集团，汇仁公司新招聘了一批新员工，现在要对新入职的员工进行培训，你现在职务就是汇仁的培训经理。'},
    #                           {'role': 'user', 'content': prompt}])

    # 船长互动案例
    system_prompt = """
    你现在是一个销售培训讲师，你的学生来培训的目标是学习沟通和销售技巧,你的任务是和你的学员进行互动问答，任务要求如下：
{
    1.和学员进行总轮数为 4 轮的互动问答，最终目的是让学员结合案例和答案学习对不同行为风格的人应该使用怎样的沟通策略。
    2.首次互动的提问为固定的问题，本次互动的第一个问题为：“ 根据以上信息，结合前面学到的内容，请您判断一下小虎的行为风格属于哪种类型？若您明天要去和他去沟通公司新品进场的事儿，您将会采取怎样的沟通策略，请具体描述。”
    3.沟通策略需要答出以下几点：1.开场白2.对方的兴趣点3.如何处理对方异议4.如何询问对方意见5.如何与对方达成共识
}
学习材料：
{
1.关系型：诚恳，耐心的引导出目标，保留弹性，给予支持，关注人际关系，澄清事实
2.社交型：亲切，友好，关注团体而非个人，重视整体而非细节，提供社交的活动，提供支持，提供发表意见的机会
3.思考型：事先准备，分析利弊错失，关注任务，系统的方式，对事不对人，一致性，有耐性
4.指导型：直接，简短，重点式的答复，关注业务及成果，强调利益，提供挑战，自由及机会，问“什么？”而非“如何？”
}
}
案例材料如下：
{
小鹰是您希望开发合作的门店经理，他对您的产品能否进入这个门店起着决定性的作用。在多次沟通和观察后，您发现小鹰是一个严谨、认真且不苟言笑的人。他与门店员工的互动中，总是保持着一丝不苟的态度，对待每个细节都十分关注。在与顾客沟通时，他的风格也显得细致入微。例如，有一次您走进门店时，看到小鹰正在向一位母亲介绍一款小儿退烧药。他不仅详细询问了小孩的年龄、体重，还仔细了解了症状和使用情况。介绍药品时，他分步骤地讲解了服用方法、剂量要求以及每个可能的注意事项，并确保顾客完全理解后才结束推荐。
您进一步了解到，小鹰对产品的安全性、效果、以及使用规范非常看重，不会被简单的市场推广或广告语打动。他偏好从科学、客观的数据中获得信息，因此他往往会要求供应商提供详细的产品背景、测试报告或相关数据。作为一名推销员，您意识到，仅仅依靠产品优点的泛泛描述可能很难引起他的兴趣。
}
标准答案：
{
指导型：
开场白：开门见山，直陈拜访目的和需占用时间，请求对方允许。
兴趣点：问题的解决方案，效益，切记离题和绕圈。
询问：要直截了当，并且告诉对方提每个问题的目的，让对方主导，每提一个建议，问：您觉得可以吗？
异议：把利弊得失摊开，大家摆观点，对方为“对事不对人“，所以不必过于担心针锋相对
立约：爱憎分明，走关系套交情效果不大。
后续服务：兑现承诺，出问题按合约办。礼多反诈，点到为止。
上级压力：保留自己的观点


思考型：
开场白：简单寒暄，不要过度嘲笑。
兴趣点：问题的解决方案，新资讯，过程、细节而非结果，提供书面材料，细细讲解一遍，他还会自己再看一遍。
询问：顺着思路往下问，不要离题，喜欢精致深刻的问题，和他一起思考，有问必答。
异议：清楚自己的缺陷和应答法。通过提供新信息、新思路改变对方的观点，但不要代替他作判断，不要否定，不要下断言，要先讲“因”再讲“果”。
立约：该签的时候会签。可用时间表催促，或说服对方暂放下一些次要问题。
后续服务：不用太多关怀，别占太多时间，如果结果与预期不符，应及时处理，解释原因，与对方一起回顾原来的思路，拿出实际行动，对方不会把责任都推给你。
上级压力：分析后果，从技术上提意见，后续挑毛病，走着瞧。
}
    """
    initial_messages = [{'role': 'system', 'content': system_prompt}]
    multi_round_dialogue_stream(messages=initial_messages)

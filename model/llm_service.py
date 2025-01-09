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
        self.model_dict = {"gpt-4o", "qwen-max", "ERNIE-4.0-8K", "gpt-4o-mini"}

    @staticmethod
    def get_config():
        with open('D:\project\huiren_nlp\llm_api_service\config.json') as config_file:
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

                # res_content = response.choices[0].message.content.strip().rstrip("<|im_end|>")

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

                # res_content = response.choices[0].message.content.strip().rstrip("<|im_end|>")

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
        :param model_name: {"gpt-4o","qwen-max"}
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
            elif model_name == "ERNIE-4.0-8K":
                return self.wenxin_response_stream(model_name=model_name, message=message)
            else:
                self.logger.warning(f"model error , {model_name} not in {self.model_dict}, default gpt-4o")
                return self.chatgpt_4o_response(model_name=model_name, message=message)


if __name__ == '__main__':
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'test'}]
    _test_mes = [{'role': 'system',
                  'content': '\n    汇仁是一家大型医药企业集团，汇仁公司新招聘了一批新员工，现在要对新入职的员工进行培训，你现在职务就是汇仁的培训经理。")\n    你作为汇仁的培训经理，任务如下：\n        {\n            1.将单元内所有的题目的“答案”做为知识点，每个题目的答案可以看做一个或者多个知识点，根据知识点来依次出题，组成一套试卷\n            2.所出的每个题目包含三个部分，分别为：题目的内容，题目的答案，题目知识点（可以是1个或者多个）\n            3.所出的题目要包含所有的知识点，也就是单元内所有答案的内容\n            4.试卷的内容由填空题和问答题构成\n            5.你所输出的题目格式有固定的要求，格式为json格式，示例如下：\n        {“result":\n        [\n        {\n            "题目1": "生成的题目1",\n            "答案1": "生成的答案1",\n            "出题点1": "知识点1",\n            "题目类型":"填空题"\n        },\n        {\n            "题目2": "生成的题目2",\n            "答案2": "生成的答案2",\n            "出题点2": "知识点1，知识点2，知识点3",\n            "题目类型":"问答题"\n        }\n        ]}\n            6.只按照格式输出json内容，不输出其他内容\n        }\n'},
                 {'role': 'user',
                  'content': "下列知识为本次要培训的内容：{这次培训内容的单元名称叫做：企业的组织架构，先进的工业产能，营销网络的分布，产品集群的划分 ，每一个单元都有对应的许多题目，答案和出题点，信息如下：[{'题目': '汇仁药业的组织架构？', '答案': '总裁分管职能板块，包含：总裁办、财务中心、人力行政中心、研发中心；\\n营销副总裁分管营销板块，包含：品牌中心、电商中心、线下营销中心、多培康营销中心、流程信息中心；\\n供应链中心副总经理，包含：战略采购中心、生产中心、质量中心、物流中心、运营办。', '出题点': '了解企业的组织架构'}, {'题目': '汇仁药业的补肾品类产品？', '答案': '补肾品类包括：肾宝合剂、肾宝片、肝肾安颗粒、六味地黄丸', '出题点': '了解企业补肾品类产品'}, {'题目': '汇仁药业的调经品类产品？', '答案': '调经品类包括:女金胶囊、阿胶当归合剂、乌鸡白凤丸', '出题点': '了解企业调经品类产品'}, {'题目': '汇仁药业其他品种产品？', '答案': '其他品种的产品：牛黄解毒片、小儿止咳糖浆、引阳索胶囊、解毒痤疮丸、复方鲜竹沥液、生脉饮（人参方）、生脉饮（党参方）、理气暖胃颗粒等，我们现有药品生产批文86件，涵盖多种常见药品剂型。当前主要生产品种20余个，涉及中药饮片加工220个品规。', '出题点': '了解企业他品种产品'}, {'题目': '汇仁药业精制饮片产品品类？', '答案': '公司精制饮片产品有：燕窝，西洋参，三七、冬虫夏草等。', '出题点': '了解企业精制饮片产品'}, {'题目': '汇仁药业高品质的创新研发成果？', '答案': '建设上海——南昌双基地研发平台，与中国中医科学院、中国医学科学院药用植物研究所、中国科学院上海药物研究所、中国药科大学等科研院所合作；\\n博士后工作站，开展创新产品开发、核心品种二次开发、制造核心技术升级等研究；\\n100余个药品批文，六类以上中药新药品种20多个，主持和参与国家级重大科技项目25项，拥有著作权95件，发明专利66件。', '出题点': '了解企业高品质的创新研发成果'}, {'题目': '汇仁药业具有高品质的供应链管理？', '答案': '供应链管理方面：\\n1、设立了人参、黄芪、枸杞子、肉苁蓉等46个重点品种的中药材种养殖繁育基地,被中国中药协会评为“中国优质道地药材基地”；\\n2、 是中国中药协会中药材种植养殖专业委员会副理事长单位；\\n3、 采取“去市场化-产地化-基地化”的供应渠道策略 和“企业-合作社-农户”为主体的供应结构，保障了供给渠道稳定。', '出题点': '了解企业高品质的供应链管理'}, {'题目': '汇仁传承工匠精神，精益求精，采用“六维度—四体系”规范管理模式实现了中药材采收？', '答案': '六维度——包装储运、产地初加工、采收采集、种植过程、道地考证和基原鉴定\\n四体系——技术标准、教育培训、监督检查和全程追溯', '出题点': '了解“六维度—四体系”规范管理模式'}, {'题目': '汇仁药业的现代化工业产能？', '答案': '工业产能方面：我们汇仁智造Ⅰ期投资约6亿元人民币，生产面积逾40000㎡，建成5座数字化车间和1座自动化立体仓库。与西门子、浙远、东杰、洛施德等著名厂家合作，按行业领先的原则整机引进医药生产或检测核心仪器设备共计200余套，实现年处理中药材4520吨，产片剂15亿片，胶囊剂20.1亿粒，颗粒剂375吨，丸剂300吨，口服液29.3亿毫升的产能规模。', '出题点': '了解企业现化代化工业产能'}, {'题目': '汇仁药业运用的智造技术？', '答案': '智造技术方面：\\n智能：斥资8000万元打造，以MES系统为中心，计划管理、自动化控制、在线质量检测、信息化管理等系统全方位支撑。\\n质控：从QMS为抓手，围绕中药提取、制剂和质量检测的关键环节，覆盖药品生产的全过程，掌控超8000个关键质量控制节点。\\n信息：借助ERP、LIMS、TPCMS等系统，实现生产计划调度、物料追踪、工艺执行、电子记录、质量监控、物流管控、能源设备管理等各模块的数据集成。\\n环保：MVR高效节能技术的运用，实现年减排二氧化碳5.2万吨和废水2万吨。行业领先的“化学吸收+生化吸收+多像级等离子协同反应”三级除臭工艺践行绿色发展观。\\n领先：是全国首家将德国西门子公司MES系统运用于中成药生产的企业。', '出题点': '了解企业智造技能'}, {'题目': '汇仁药业的营销系统？', '答案': '营销系统：\\n1、线下营销中心：  10个大区51个办事处；\\n2、电商中心：品牌自营部、渠道分销部、直播部、客服部等；\\n3、多培康营销中心：28个省区', '出题点': '了解企业营销系统的组建'}, {'题目': '汇仁药业销售网络布局？', '答案': '线下营销中心渠道布局覆盖全国10   个大区，51个办事处\\n西北大区（甘肃/新疆/陕西）3  办事处；\\n川渝大区（成都/乐山/重庆/南充）4  办事处；\\n西南大区（贵州/柳州/南宁/云南1和2）5  办事处；\\n华南大区(湛江/佛山/深圳/广州/东莞)  5  办事处；\\n中南大区(福州/泉州/长沙/怀化/江西)5  办事处；\\n东南大区(杭州/温州/金华/上海/宁波) 5  办事处；\\n华中大区(襄阳/武汉/郑州/焦作/洛阳/合肥/芜湖)7  办事处；\\n华东大区(济南/潍坊/临沂/青岛/南通/苏州/淮安/南京)8   办事处；\\n东北大区（哈尔滨/吉林/锦州/沈阳）4 办事处；\\n华北大区（太原/北京/天津/石家庄/呼市）5  办事处。', '出题点': '了解企业销售网络'}, {'题目': '汇仁药业所获得过的荣誉？', '答案': '截至2023年12月，汇仁荣获县级以上各类表彰、奖励共计379项次，其中省级奖项85次，国家级奖项24次。', '出题点': '了解企业荣誉'}, {'题目': '汇仁药业未来的事业版图？', '答案': '未来的事业版图，我们主要分为一主两翼，一主是补肾类产品集群；两翼是女性产品集群和药食同源产品集群', '出题点': '了解企业未来发展版图'}, {'题目': '汇仁药业如何打造补肾类产品集群？', '答案': '在药业，我们要继续：\\n1、做大做强男性和补益补类产品 ，这是我们做蛋糕的板块，要继续抓住用户的心智\\n2、要拓展保健品领域，打造引流品\\n3、通过线上+线下渠道并重的方式让药业成为龙头', '出题点': '了解企业未来产品集群发展规划'}, {'题目': '汇仁药业如何打造女性产品集群？', '答案': '在电商板块，我们要继续：\\n1、承接汇仁品牌的影响力做好补益类产品，同时在女性品类上发力\\n2、要拓展保健品领域，打造引流品\\n3、通过线上渠道为主，线下渠道为辅（线上爆品，线下引流）的发展方式成为汇仁另一大拼图板块。', '出题点': '了解企业未来产品集群发展规划'}, {'题目': '汇仁药业如何打造药食同源产品集群？', '答案': '在汇仁多培康、电商、饮片板块，我们要继续：\\n1、承接汇仁品牌的影响力做好补益类产品，同时在维矿类、普药和儿童/青少年品类重点抢市场上的蛋糕，同步做一定的蛋糕\\n2、要拓展保健品领域，打造引流品\\n3、通过线下渠道为主，线上渠道为辅的发展方式成为汇仁另一大拼图板块。', '出题点': '了解企业未来产品集群发展规划'}, {'题目': '汇仁药业中长期发展规划？', '答案': '第一阶段（3-5年）50亿销售规模上市完成，消费类药品：品牌+明星品 驱动；通过电商测试+渠道验证，以扩大GMV 为目的，收购相关批文。\\n第二阶段（6-10年）100亿销售规模，消费类为主+院线类为辅：通过资本收购相关产业，赋能消费类药品 再增长；通过资本收购院线类药品企业相关资产，嫁接企业基因，提供可持续增长空间；新药首仿；拥有产品批文矩阵。\\n第三阶段（10+年）200亿+销售规模，消费类+院线类药品并重开始投入原研，提升品牌力：实现渠道相互赋能、药品相互赋能；中国医药企业，必将投入原研，这是我们 这代人的行业使命，为国力、民族自信注 入力量。前提是做好销售、做好品牌。', '出题点': '了解企业中长期发展规划'}]}"}]

    wenxin_test = [{'role': 'user',
                    'content': """\n    汇仁是一家大型医药企业集团，汇仁公司新招聘了一批新员工，现在要对新入职的员工进行培训，你现在职务就是汇仁的培训经理。")\n    你作为汇仁的培训经理，任务如下：\n        {\n            1.将单元内所有的题目的“答案”做为知识点，每个题目的答案可以看做一个或者多个知识点，根据知识点来依次出题，组成一套试卷\n            2.所出的每个题目包含三个部分，分别为：题目的内容，题目的答案，题目知识点（可以是1个或者多个）\n            3.所出的题目要包含所有的知识点，也就是单元内所有答案的内容\n            4.试卷的内容由填空题和问答题构成\n            5.你所输出的题目格式有固定的要求，格式为json格式，示例如下：\n        {“result":\n        [\n        {\n            "题目1": "生成的题目1",\n            "答案1": "生成的答案1",\n            "出题点1": "知识点1",\n            "题目类型":"填空题"\n        },\n        {\n            "题目2": "生成的题目2",\n            "答案2": "生成的答案2",\n            "出题点2": "知识点1，知识点2，知识点3",\n            "题目类型":"问答题"\n        }\n        ]}\n            6.只按照格式输出json内容，不输出其他内容\n        }\n 下列知识为本次要培训的内容：{这次培训内容的单元名称叫做：企业的组织架构，先进的工业产能，营销网络的分布，产品集群的划分 ，每一个单元都有对应的许多题目，答案和出题点，信息如下：[{'题目': '汇仁药业的组织架构？', '答案': '总裁分管职能板块，包含：总裁办、财务中心、人力行政中心、研发中心；\\n营销副总裁分管营销板块，包含：品牌中心、电商中心、线下营销中心、多培康营销中心、流程信息中心；\\n供应链中心副总经理，包含：战略采购中心、生产中心、质量中心、物流中心、运营办。', '出题点': '了解企业的组织架构'}, {'题目': '汇仁药业的补肾品类产品？', '答案': '补肾品类包括：肾宝合剂、肾宝片、肝肾安颗粒、六味地黄丸', '出题点': '了解企业补肾品类产品'}, {'题目': '汇仁药业的调经品类产品？', '答案': '调经品类包括:女金胶囊、阿胶当归合剂、乌鸡白凤丸', '出题点': '了解企业调经品类产品'}, {'题目': '汇仁药业其他品种产品？', '答案': '其他品种的产品：牛黄解毒片、小儿止咳糖浆、引阳索胶囊、解毒痤疮丸、复方鲜竹沥液、生脉饮（人参方）、生脉饮（党参方）、理气暖胃颗粒等，我们现有药品生产批文86件，涵盖多种常见药品剂型。当前主要生产品种20余个，涉及中药饮片加工220个品规。', '出题点': '了解企业他品种产品'}, {'题目': '汇仁药业精制饮片产品品类？', '答案': '公司精制饮片产品有：燕窝，西洋参，三七、冬虫夏草等。', '出题点': '了解企业精制饮片产品'}, {'题目': '汇仁药业高品质的创新研发成果？', '答案': '建设上海——南昌双基地研发平台，与中国中医科学院、中国医学科学院药用植物研究所、中国科学院上海药物研究所、中国药科大学等科研院所合作；\\n博士后工作站，开展创新产品开发、核心品种二次开发、制造核心技术升级等研究；\\n100余个药品批文，六类以上中药新药品种20多个，主持和参与国家级重大科技项目25项，拥有著作权95件，发明专利66件。', '出题点': '了解企业高品质的创新研发成果'}, {'题目': '汇仁药业具有高品质的供应链管理？', '答案': '供应链管理方面：\\n1、设立了人参、黄芪、枸杞子、肉苁蓉等46个重点品种的中药材种养殖繁育基地,被中国中药协会评为“中国优质道地药材基地”；\\n2、 是中国中药协会中药材种植养殖专业委员会副理事长单位；\\n3、 采取“去市场化-产地化-基地化”的供应渠道策略 和“企业-合作社-农户”为主体的供应结构，保障了供给渠道稳定。', '出题点': '了解企业高品质的供应链管理'}, {'题目': '汇仁传承工匠精神，精益求精，采用“六维度—四体系”规范管理模式实现了中药材采收？', '答案': '六维度——包装储运、产地初加工、采收采集、种植过程、道地考证和基原鉴定\\n四体系——技术标准、教育培训、监督检查和全程追溯', '出题点': '了解“六维度—四体系”规范管理模式'}, {'题目': '汇仁药业的现代化工业产能？', '答案': '工业产能方面：我们汇仁智造Ⅰ期投资约6亿元人民币，生产面积逾40000㎡，建成5座数字化车间和1座自动化立体仓库。与西门子、浙远、东杰、洛施德等著名厂家合作，按行业领先的原则整机引进医药生产或检测核心仪器设备共计200余套，实现年处理中药材4520吨，产片剂15亿片，胶囊剂20.1亿粒，颗粒剂375吨，丸剂300吨，口服液29.3亿毫升的产能规模。', '出题点': '了解企业现化代化工业产能'}, {'题目': '汇仁药业运用的智造技术？', '答案': '智造技术方面：\\n智能：斥资8000万元打造，以MES系统为中心，计划管理、自动化控制、在线质量检测、信息化管理等系统全方位支撑。\\n质控：从QMS为抓手，围绕中药提取、制剂和质量检测的关键环节，覆盖药品生产的全过程，掌控超8000个关键质量控制节点。\\n信息：借助ERP、LIMS、TPCMS等系统，实现生产计划调度、物料追踪、工艺执行、电子记录、质量监控、物流管控、能源设备管理等各模块的数据集成。\\n环保：MVR高效节能技术的运用，实现年减排二氧化碳5.2万吨和废水2万吨。行业领先的“化学吸收+生化吸收+多像级等离子协同反应”三级除臭工艺践行绿色发展观。\\n领先：是全国首家将德国西门子公司MES系统运用于中成药生产的企业。', '出题点': '了解企业智造技能'}, {'题目': '汇仁药业的营销系统？', '答案': '营销系统：\\n1、线下营销中心：  10个大区51个办事处；\\n2、电商中心：品牌自营部、渠道分销部、直播部、客服部等；\\n3、多培康营销中心：28个省区', '出题点': '了解企业营销系统的组建'}, {'题目': '汇仁药业销售网络布局？', '答案': '线下营销中心渠道布局覆盖全国10   个大区，51个办事处\\n西北大区（甘肃/新疆/陕西）3  办事处；\\n川渝大区（成都/乐山/重庆/南充）4  办事处；\\n西南大区（贵州/柳州/南宁/云南1和2）5  办事处；\\n华南大区(湛江/佛山/深圳/广州/东莞)  5  办事处；\\n中南大区(福州/泉州/长沙/怀化/江西)5  办事处；\\n东南大区(杭州/温州/金华/上海/宁波) 5  办事处；\\n华中大区(襄阳/武汉/郑州/焦作/洛阳/合肥/芜湖)7  办事处；\\n华东大区(济南/潍坊/临沂/青岛/南通/苏州/淮安/南京)8   办事处；\\n东北大区（哈尔滨/吉林/锦州/沈阳）4 办事处；\\n华北大区（太原/北京/天津/石家庄/呼市）5  办事处。', '出题点': '了解企业销售网络'}, {'题目': '汇仁药业所获得过的荣誉？', '答案': '截至2023年12月，汇仁荣获县级以上各类表彰、奖励共计379项次，其中省级奖项85次，国家级奖项24次。', '出题点': '了解企业荣誉'}, {'题目': '汇仁药业未来的事业版图？', '答案': '未来的事业版图，我们主要分为一主两翼，一主是补肾类产品集群；两翼是女性产品集群和药食同源产品集群', '出题点': '了解企业未来发展版图'}, {'题目': '汇仁药业如何打造补肾类产品集群？', '答案': '在药业，我们要继续：\\n1、做大做强男性和补益补类产品 ，这是我们做蛋糕的板块，要继续抓住用户的心智\\n2、要拓展保健品领域，打造引流品\\n3、通过线上+线下渠道并重的方式让药业成为龙头', '出题点': '了解企业未来产品集群发展规划'}, {'题目': '汇仁药业如何打造女性产品集群？', '答案': '在电商板块，我们要继续：\\n1、承接汇仁品牌的影响力做好补益类产品，同时在女性品类上发力\\n2、要拓展保健品领域，打造引流品\\n3、通过线上渠道为主，线下渠道为辅（线上爆品，线下引流）的发展方式成为汇仁另一大拼图板块。', '出题点': '了解企业未来产品集群发展规划'}, {'题目': '汇仁药业如何打造药食同源产品集群？', '答案': '在汇仁多培康、电商、饮片板块，我们要继续：\\n1、承接汇仁品牌的影响力做好补益类产品，同时在维矿类、普药和儿童/青少年品类重点抢市场上的蛋糕，同步做一定的蛋糕\\n2、要拓展保健品领域，打造引流品\\n3、通过线下渠道为主，线上渠道为辅的发展方式成为汇仁另一大拼图板块。', '出题点': '了解企业未来产品集群发展规划'}, {'题目': '汇仁药业中长期发展规划？', '答案': '第一阶段（3-5年）50亿销售规模上市完成，消费类药品：品牌+明星品 驱动；通过电商测试+渠道验证，以扩大GMV 为目的，收购相关批文。\\n第二阶段（6-10年）100亿销售规模，消费类为主+院线类为辅：通过资本收购相关产业，赋能消费类药品 再增长；通过资本收购院线类药品企业相关资产，嫁接企业基因，提供可持续增长空间；新药首仿；拥有产品批文矩阵。\\n第三阶段（10+年）200亿+销售规模，消费类+院线类药品并重开始投入原研，提升品牌力：实现渠道相互赋能、药品相互赋能；中国医药企业，必将投入原研，这是我们 这代人的行业使命，为国力、民族自信注 入力量。前提是做好销售、做好品牌。', '出题点': '了解企业中长期发展规划"""}]
    logger = CustomLogger("HuiRen nlp Logger")
    _test = LLMService(llm_logger=logger)
    response = _test.get_response_stream(model_name="ERNIE-4.0-8K", messages=wenxin_test)
    for chunk in response:
        try:
            if chunk['body']['is_end'] is True:
                print("[END]")
            else:
                chunk_content = f"data: {chunk['body']['result']}\n\n"
                print(chunk_content)
        except Exception as e:
            print("data: [END]\n\n")
            logger.info(f"yield [END], wenxin chunk is None, {e}")

    # for chunk in response:
    #     if (len(chunk.choices)) == 0:
    #         print("13132131312")
    #         continue
    #     if chunk.choices[0].delta.content == None:
    #         print("13132131312222222222222222222")
    #         continue
    #     print(chunk.choices[0].delta.content)

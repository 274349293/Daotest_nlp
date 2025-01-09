from fastapi import FastAPI
from llm_api_service.ques_judg import ques_judgment, QjInfo
from llm_api_service.behavioral_style_test import behavioral_style, BehavioralStyleInfo

app = FastAPI()

"""
接口集成
1. ques_judgment 选择题判断题判断接口
2. behavioral style test 学员行为风格测试获取结果接口(视频流中的互动题)

"""


@app.post("/ques_judgment")
def ques_judgment_fun(qj_info: QjInfo):
    answer = ques_judgment(qj_info)
    return answer


@app.post("/behavioral_style_test")
def behavioral_style_test_fun(behavioral_style_info: BehavioralStyleInfo):
    test_res = behavioral_style(behavioral_style_info)
    return test_res


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)

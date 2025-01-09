from fastapi import FastAPI
from llm_api_service.ques_judg import ques_judgment, QjInfo
from llm_api_service.behavioral_style_test import behavioral_style, BehavioralStyleInfo
from llm_api_service.exam_marking import exam_mark, ExamQaInfo
from llm_api_service.practice_stream import get_stream_response, PracticeQaInfo

app = FastAPI()

"""
接口集成
1. ques_judgment 选择题判断题判断接口
2. behavioral_style_test 学员行为风格测试获取结果接口(视频流中的互动题)
3. exam_mark 通用考试评价接口(简答题和阅读理解题)
4. practice_stream 练习评价接口，流式返回
"""


@app.post("/ques_judgment")
def ques_judgment_fun(qj_info: QjInfo):
    answer = ques_judgment(qj_info)
    return answer


@app.post("/behavioral_style_test")
def behavioral_style_test_fun(behavioral_style_info: BehavioralStyleInfo):
    test_res = behavioral_style(behavioral_style_info)
    return test_res


@app.post("/exam_mark")
def exam_mark_fun(qa_info: ExamQaInfo):
    exam_res = exam_mark(qa_info)
    return exam_res


@app.post("/practice_stream")
def practice_stream_fun(qa_info: PracticeQaInfo):
    practice_res = get_stream_response(qa_info)
    return practice_res


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)

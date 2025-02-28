from fastapi import FastAPI, BackgroundTasks, HTTPException
from llm_api_service.ques_judg import ques_judgment, QjInfo
from llm_api_service.behavioral_style_test import behavioral_style, BehavioralStyleInfo
from llm_api_service.exam_marking import exam_mark, ExamQaInfo
from llm_api_service.practice_stream import get_stream_response, PracticeQaInfo
from llm_api_service.multi_round_dialogue import multi_round_dialogue, DialogueInfo
from llm_api_service.multi_round_dialogue_mark import multi_round_dialogue_mark, DialogueMarkInfo
from llm_api_service.tag_generation import tag_generation, TagSet
from llm_api_service.qa_generation import process_qa_generation, QaGeneration

app = FastAPI()

"""
接口集成
1. ques_judgment 选择题判断题判断接口
2. behavioral_style_test 学员行为风格测试获取结果接口(视频流中的互动题)
3. exam_mark 通用考试评价接口(简答题和阅读理解题)
4. practice_stream 练习评价接口，流式返回
5. mr_dialogue 案例分析题接口，课后互动题-多轮对话 
6. tag_generation 标签生成接口
7. qa_generation 题目生成接口
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
async def practice_stream_fun(qa_info: PracticeQaInfo):
    return await get_stream_response(qa_info)


@app.post("/mr_dialogue")
async def multi_round_dialogue_fun(dialogue_info: DialogueInfo):
    return await multi_round_dialogue(dialogue_info)


@app.post("/mr_dialogue_mark")
def multi_round_dialogue_mark_fun(dialogue_mark_info: DialogueMarkInfo):
    return multi_round_dialogue_mark(dialogue_mark_info)


@app.post("/tag_generation")
def tag_generation_fun(tag_set: TagSet):
    qa_res = tag_generation(tag_set)
    return qa_res


@app.post("/qa_generation")
async def qa_generation_fun(qa_gen: QaGeneration, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_qa_generation, qa_gen)
    return {"status": 1}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)

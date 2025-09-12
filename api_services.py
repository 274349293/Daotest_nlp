from fastapi import FastAPI, BackgroundTasks, HTTPException
from llm_api_service.ques_judg import ques_judgment, QjInfo
from llm_api_service.behavioral_style_test import behavioral_style, BehavioralStyleInfo
from llm_api_service.exam_marking import exam_mark, ExamQaInfo
from llm_api_service.practice_stream import get_stream_response, PracticeQaInfo
from llm_api_service.multi_round_dialogue import multi_round_dialogue, DialogueInfo
from llm_api_service.multi_round_dialogue_mark import multi_round_dialogue_mark, DialogueMarkInfo
from llm_api_service.tag_generation import tag_generation, TagSet
from llm_api_service.qa_generation import process_qa_generation, process_convenient_qa_generation, QaGeneration
from llm_api_service.decompose_knowledge_point import decompose_knowledge_point, KnowledgePoint
from llm_api_service.azure_realtime_function_call import realtime_function_call, RealtimeFunctionCallInfo
from llm_api_service.llm_chat import optimized_multi_round_dialogue, OptimizedDialogueInfo
from llm_api_service.chat_rating import chat_rating, ChatRatingInfo
from llm_api_service.golf_llm import golf_llm_chat, GolfLLMInfo
from llm_api_service.three_point_las_vegas import calculate_lasi_score, LaSiGameData
from llm_api_service.three_point_las_vegas_simple import calculate_tee_order, LaSiGameData
from llm_api_service.stroke_and_match_combo import calculate_stroke_match_score, StrokeMatchGameData
from llm_api_service.golf_combined_scoring import golf_combined_scoring, GolfCombinedGameData

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
8. decompose_knowledge 将整段的知识点拆分成若干个相对独立的知识点
9. realtime_function_call 实时语音模型function call 结果返回(关键词检索）
10. realtime_function_call_advanced 实时语音模型function call 结果返回(用户query检索）
11. llm_chat 优化后的多轮对话接口，支持不同场景
12. golf_llm 高尔夫相关问题的智能回复接口
13. three_point_las_vegas 高尔夫游戏 拉丝3点
14. stroke_match_combo 高尔夫游戏 比杆比洞
15. tee_order_calculation 高尔夫游戏 击球顺序计算（临时接口）
16. golf_combined_score 高尔夫综合计分接口（整合拉丝3点和挂杆挂洞）[临时接口]
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
    if qa_gen.type == 1:
        background_tasks.add_task(process_qa_generation, qa_gen)
    elif qa_gen.type == 0:
        background_tasks.add_task(process_convenient_qa_generation, qa_gen)
    else:
        return {"status": 0}
    return {"status": 1}


@app.post("/decompose_knowledge")
def decompose_knowledge_fun(kg_p: KnowledgePoint):
    res = decompose_knowledge_point(kg_p)
    return res


@app.post("/realtime_function_call")
async def realtime_function_call_fun(fc_info: RealtimeFunctionCallInfo):
    return await realtime_function_call(fc_info, "full_content")


@app.post("/realtime_function_call_advanced")
async def realtime_function_call_advanced_fun(fc_info: RealtimeFunctionCallInfo):
    return await realtime_function_call(fc_info, "advanced_retrieval")


@app.post("/llm_chat")
async def llm_chat_fun(dialogue_info: OptimizedDialogueInfo):
    return await optimized_multi_round_dialogue(dialogue_info)


@app.post("/chat_rating")
def chat_rating_fun(rating_info: ChatRatingInfo):
    result = chat_rating(rating_info)
    return result


@app.post("/golf_llm")
async def golf_llm_fun(golf_info: GolfLLMInfo):
    return await golf_llm_chat(golf_info)


@app.post("/lasi_scoring")
def lasi_scoring_fun(game_data: LaSiGameData):
    result = calculate_lasi_score(game_data)
    return result


@app.post("/stroke_match_combo")
def stroke_match_combo_fun(game_data: StrokeMatchGameData):
    result = calculate_stroke_match_score(game_data)
    return result


# 在所有接口定义的最后添加新的接口：

@app.post("/tee_order_calculation")
def tee_order_calculation_fun(game_data: LaSiGameData):
    """
    拉丝三点高尔夫击球顺序计算接口

    功能：
    - 计算乱拉模式下每洞的击球顺序
    - 基于上一洞净杆数排序
    - 应用让杆规则和让杆限制条件
    - 填充并返回完整的tee_order数据

    参数：
    - game_data: LaSiGameData - 游戏数据，第一洞tee_order需完整，其他洞tee_order可为空

    返回：
    - LaSiGameData - 填充好所有洞tee_order的完整数据
    """
    try:
        result = calculate_tee_order(game_data)
        return result
    except Exception as e:
        # 可以根据需要返回更详细的错误信息
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/golf_combined_score")
def golf_combined_score_fun(combined_data: GolfCombinedGameData):
    """
    高尔夫综合计分接口

    功能：
    - 整合拉丝3点和挂杆挂洞两种比赛模式的计分
    - 返回每个选手在两种比赛中的得分汇总
    - 计算每个选手的总分

    参数：
    - combined_data: GolfCombinedGameData - 包含两种比赛数据的综合对象

    返回：
    - Dict - 包含每个选手分数汇总的结果
    """
    try:
        result = golf_combined_scoring(combined_data)
        return result
    except Exception as e:
        # 直接抛异常，按需求要求
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)

from fastapi import FastAPI
from llm_api_service.ques_judg import ques_judgment, QjInfo

app = FastAPI()

"""
接口集成
1. ques_judgment 选择题判断题判断接口
2. 

"""


@app.post("/ques_judgment")
def ques_judgment_fun(qj_info: QjInfo):
    answer = ques_judgment(qj_info)
    return answer


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)

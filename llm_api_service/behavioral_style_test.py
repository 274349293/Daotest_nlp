from fastapi.responses import StreamingResponse
import ast
import json
from fastapi import FastAPI
from model.llm_service import LLMService
from pydantic import BaseModel
from utils.nlp_logging import CustomLogger

logger = CustomLogger(name="HuiRen behavioral style test api", write_to_file=True)
llm = LLMService(llm_logger=logger)


class BehavioralStyleInfo(BaseModel):
    A1: int
    B1: int
    A2: int
    B2: int
    A3: int
    B3: int
    A4: int
    B4: int
    A5: int
    B5: int
    A6: int
    B6: int
    A7: int
    B7: int
    A8: int
    B8: int
    A9: int
    B9: int
    A10: int
    B10: int
    A11: int
    B11: int
    A12: int
    B12: int
    A13: int
    B13: int
    A14: int
    B14: int
    A15: int
    B15: int
    A16: int
    B16: int
    A17: int
    B17: int
    A18: int
    B18: int


def behavioral_style_score_calculate(behavioral_style_info: BehavioralStyleInfo):
    o_score = (behavioral_style_info.A1 + behavioral_style_info.B3 + behavioral_style_info.A5 + behavioral_style_info.B7
               + behavioral_style_info.A9 + behavioral_style_info.B11 + behavioral_style_info.A13 +
               behavioral_style_info.B15 + behavioral_style_info.A17)
    s_score = (behavioral_style_info.B1 + behavioral_style_info.A3 + behavioral_style_info.B5 + behavioral_style_info.A7
               + behavioral_style_info.B9 + behavioral_style_info.A11 + behavioral_style_info.B13 +
               behavioral_style_info.A15 + behavioral_style_info.B17)
    d_score = (behavioral_style_info.B2 + behavioral_style_info.A4 + behavioral_style_info.B6 + behavioral_style_info.A8
               + behavioral_style_info.B10 + behavioral_style_info.A12 + behavioral_style_info.B14 +
               behavioral_style_info.A16 + behavioral_style_info.B18)
    i_score = (behavioral_style_info.A2 + behavioral_style_info.B4 + behavioral_style_info.A6 + behavioral_style_info.B8
               + behavioral_style_info.A10 + behavioral_style_info.B12 + behavioral_style_info.A14 +
               behavioral_style_info.B16 + behavioral_style_info.A18)
    if o_score > s_score and d_score > i_score:
        return "社交型"
    elif o_score > s_score and i_score > d_score:
        return "关系型"
    elif s_score > o_score and d_score > i_score:
        return "指导型"
    elif s_score > o_score and i_score > d_score:
        return "思考型"
    else:
        logger.error(f"behavioral style info is {behavioral_style_info}")
        return None


def behavioral_style(behavioral_style_info: BehavioralStyleInfo):
    logger.info("------------------start--------------------")
    logger.info(behavioral_style_info)
    try:
        result = behavioral_style_score_calculate(behavioral_style_info)
        if result is not None:
            logger.info(f"behavioral style test result is {result}")
            return {"result": result}
        else:
            logger.error(f"behavioral style test score calculate error, calculate fun return None")
            return "error"
    except Exception as e:
        logger.error(f"behavioral_style error: {e}")
        return "error"

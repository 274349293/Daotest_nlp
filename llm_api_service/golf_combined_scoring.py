from typing import Dict, Any
from pydantic import BaseModel
from utils.nlp_logging import CustomLogger
from .three_point_las_vegas import calculate_lasi_score, LaSiGameData
from .stroke_and_match_combo import calculate_stroke_match_score, StrokeMatchGameData

"""
高尔夫综合计分接口
整合拉丝3点和挂杆挂洞两种比赛模式的计分结果
"""

logger = CustomLogger(name="Golf Combined Scoring API", write_to_file=True)


class GolfCombinedGameData(BaseModel):
    """高尔夫综合比赛数据"""
    拉丝3点: LaSiGameData
    挂杆挂洞: StrokeMatchGameData


def golf_combined_scoring(combined_data: GolfCombinedGameData) -> Dict[str, Any]:
    """
    高尔夫综合计分主函数

    Args:
        combined_data: 包含两种比赛模式数据的综合数据

    Returns:
        Dict: 每个选手在两种比赛中的得分汇总

    Raises:
        Exception: 当任一比赛计分失败时抛出异常
    """
    logger.info("------------------高尔夫综合计分开始--------------------")

    try:
        # 1. 计算拉丝3点得分
        logger.info("开始计算拉丝3点比赛得分...")
        lasi_result = calculate_lasi_score(combined_data.拉丝3点)

        if not lasi_result.get("success"):
            error_msg = f"拉丝3点计分失败: {lasi_result.get('message', '未知错误')}"
            logger.error(error_msg)
            raise Exception(error_msg)

        logger.info("拉丝3点计分成功")

        # 2. 计算挂杆挂洞得分
        logger.info("开始计算挂杆挂洞比赛得分...")
        stroke_match_result = calculate_stroke_match_score(combined_data.挂杆挂洞)

        if not stroke_match_result.get("success"):
            error_msg = f"挂杆挂洞计分失败: {stroke_match_result.get('message', '未知错误')}"
            logger.error(error_msg)
            raise Exception(error_msg)

        logger.info("挂杆挂洞计分成功")

        # 3. 提取两种比赛的选手得分
        lasi_scores = _extract_lasi_scores(lasi_result)
        stroke_match_scores = _extract_stroke_match_scores(stroke_match_result)

        # 4. 合并计分结果
        combined_scores = _merge_scores(lasi_scores, stroke_match_scores)

        logger.info("高尔夫综合计分成功完成")
        logger.info(f"最终结果: {combined_scores}")

        return {
            "success": True,
            "data": combined_scores,
            "message": "综合计分成功"
        }

    except Exception as e:
        logger.error(f"高尔夫综合计分失败: {str(e)}")
        raise


def _extract_lasi_scores(lasi_result: Dict[str, Any]) -> Dict[str, float]:
    """从拉丝3点结果中提取选手得分"""
    scores = {}

    try:
        final_scores = lasi_result["data"]["game_summary"]["final_scores"]

        for score_info in final_scores:
            player_name = score_info["player_name"]
            # 使用最终调整后的分数
            final_score = score_info["after_adjustment"]
            scores[player_name] = final_score

        logger.info(f"拉丝3点得分提取成功: {scores}")
        return scores

    except (KeyError, TypeError) as e:
        error_msg = f"拉丝3点得分提取失败: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def _extract_stroke_match_scores(stroke_match_result: Dict[str, Any]) -> Dict[str, float]:
    """从挂杆挂洞结果中提取选手得分"""
    scores = {}

    try:
        final_scores = stroke_match_result["data"]["game_summary"]["final_scores"]

        for score_info in final_scores:
            player_name = score_info["player_name"]
            total_score = score_info["total_score"]
            scores[player_name] = total_score

        logger.info(f"挂杆挂洞得分提取成功: {scores}")
        return scores

    except (KeyError, TypeError) as e:
        error_msg = f"挂杆挂洞得分提取失败: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def _merge_scores(lasi_scores: Dict[str, float],
                  stroke_match_scores: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """合并两种比赛的得分"""

    # 获取所有选手名单
    all_players = set(lasi_scores.keys()) | set(stroke_match_scores.keys())

    if not all_players:
        raise Exception("未找到任何选手数据")

    # 检查选手名单是否一致
    if set(lasi_scores.keys()) != set(stroke_match_scores.keys()):
        logger.warning("两种比赛的选手名单不完全一致")
        logger.warning(f"拉丝3点选手: {list(lasi_scores.keys())}")
        logger.warning(f"挂杆挂洞选手: {list(stroke_match_scores.keys())}")

    combined_result = {}

    for player in all_players:
        lasi_score = lasi_scores.get(player, 0)
        stroke_match_score = stroke_match_scores.get(player, 0)
        total_score = lasi_score + stroke_match_score

        combined_result[player] = {
            "拉丝3点": lasi_score,
            "挂杆挂洞": stroke_match_score,
            "总计": total_score
        }

    logger.info(f"得分合并完成，共{len(combined_result)}名选手")
    return combined_result

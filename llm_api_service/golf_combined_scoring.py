from typing import Dict, Any, List
from pydantic import BaseModel
from utils.nlp_logging import CustomLogger
from .three_point_las_vegas import calculate_lasi_score, LaSiGameData
from .stroke_and_match_combo import calculate_stroke_match_score, StrokeMatchGameData

"""
高尔夫综合计分接口
整合拉丝3点和挂杆挂洞两种比赛模式的计分结果
支持动态数据结构，根据rule字段自动识别算法类型
"""

logger = CustomLogger(name="Golf Combined Scoring API", write_to_file=True)


def golf_combined_scoring(combined_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    高尔夫综合计分主函数 - 支持动态数据结构

    Args:
        combined_data: 包含多个比赛数据块的字典，每个数据块根据rule字段确定算法类型

    Returns:
        Dict: 每个选手在各种比赛中的得分汇总

    Raises:
        Exception: 当任一比赛计分失败时抛出异常
    """
    logger.info("------------------高尔夫综合计分开始--------------------")

    try:
        # 1. 分类处理各个比赛数据块
        lasi_results = []
        stroke_match_results = []

        # 遍历最外层所有key
        for key, game_data in combined_data.items():
            logger.info(f"处理比赛数据块: {key}")

            if not isinstance(game_data, dict) or "rule" not in game_data:
                logger.warning(f"跳过无效数据块: {key} (缺少rule字段)")
                continue

            rule = game_data["rule"]
            logger.info(f"数据块 {key} 的规则类型: {rule}")

            if rule == "拉丝3点":
                # 转换为拉丝3点数据格式并计算
                lasi_data = _convert_to_lasi_data(game_data, key)
                lasi_result = calculate_lasi_score(lasi_data)

                if not lasi_result.get("success"):
                    error_msg = f"拉丝3点计分失败 ({key}): {lasi_result.get('message', '未知错误')}"
                    logger.error(error_msg)
                    raise Exception(error_msg)

                lasi_results.append({"key": key, "result": lasi_result})
                logger.info(f"拉丝3点计分成功: {key}")

            elif rule == "挂杆挂洞":
                # 转换为挂杆挂洞数据格式并计算
                stroke_match_data = _convert_to_stroke_match_data(game_data, key)
                stroke_match_result = calculate_stroke_match_score(stroke_match_data)

                if not stroke_match_result.get("success"):
                    error_msg = f"挂杆挂洞计分失败 ({key}): {stroke_match_result.get('message', '未知错误')}"
                    logger.error(error_msg)
                    raise Exception(error_msg)

                stroke_match_results.append({"key": key, "result": stroke_match_result})
                logger.info(f"挂杆挂洞计分成功: {key}")

            else:
                logger.warning(f"未知的规则类型: {rule} (数据块: {key})")
                continue

        # 2. 汇总所有比赛的选手得分
        combined_scores = _merge_all_scores(lasi_results, stroke_match_results)

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


def _convert_to_lasi_data(game_data: Dict[str, Any], key: str) -> LaSiGameData:
    """将动态数据转换为拉丝3点数据格式"""
    try:
        lasi_data = LaSiGameData(
            game_config=game_data["game_config"],
            players=game_data["players"],
            holes=game_data["holes"]
        )
        logger.info(f"拉丝3点数据转换成功: {key}")
        return lasi_data
    except Exception as e:
        error_msg = f"拉丝3点数据转换失败 ({key}): {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def _convert_to_stroke_match_data(game_data: Dict[str, Any], key: str) -> StrokeMatchGameData:
    """将动态数据转换为挂杆挂洞数据格式"""
    try:
        stroke_match_data = StrokeMatchGameData(
            game_config=game_data["game_config"],
            players=game_data["players"],
            holes=game_data["holes"]
        )
        logger.info(f"挂杆挂洞数据转换成功: {key}")
        return stroke_match_data
    except Exception as e:
        error_msg = f"挂杆挂洞数据转换失败 ({key}): {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def _merge_all_scores(lasi_results: List[Dict[str, Any]],
                      stroke_match_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """合并所有比赛的得分"""

    # 收集所有选手和他们的得分
    player_scores = {}

    # 处理拉丝3点结果
    for lasi_item in lasi_results:
        key = lasi_item["key"]
        result = lasi_item["result"]

        try:
            final_scores = result["data"]["game_summary"]["final_scores"]

            for score_info in final_scores:
                player_name = score_info["player_name"]
                # 使用最终调整后的分数
                final_score = score_info["after_adjustment"]

                # 初始化选手数据
                if player_name not in player_scores:
                    player_scores[player_name] = {}

                # 记录拉丝3点得分（使用数据块的key作为标识）
                player_scores[player_name][f"拉丝3点_{key}"] = final_score

            logger.info(f"拉丝3点得分提取成功 ({key})")

        except (KeyError, TypeError) as e:
            error_msg = f"拉丝3点得分提取失败 ({key}): {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    # 处理挂杆挂洞结果
    for stroke_match_item in stroke_match_results:
        key = stroke_match_item["key"]
        result = stroke_match_item["result"]

        try:
            final_scores = result["data"]["game_summary"]["final_scores"]

            for score_info in final_scores:
                player_name = score_info["player_name"]
                total_score = score_info["total_score"]

                # 初始化选手数据
                if player_name not in player_scores:
                    player_scores[player_name] = {}

                # 记录挂杆挂洞得分（使用数据块的key作为标识）
                player_scores[player_name][f"挂杆挂洞_{key}"] = total_score

            logger.info(f"挂杆挂洞得分提取成功 ({key})")

        except (KeyError, TypeError) as e:
            error_msg = f"挂杆挂洞得分提取失败 ({key}): {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    # 计算每个选手的总分
    final_result = {}
    for player_name, scores in player_scores.items():
        total_score = sum(scores.values())

        final_result[player_name] = {
            **scores,  # 包含各个比赛的详细得分
            "总计": total_score
        }

    logger.info(f"得分合并完成，共{len(final_result)}名选手")
    return final_result
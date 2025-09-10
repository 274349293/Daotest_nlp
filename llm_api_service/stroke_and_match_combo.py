import json
import copy
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from utils.nlp_logging import CustomLogger

"""
比杆比洞（挂杆挂洞）高尔夫游戏计分接口
支持三种游戏模式：
1. 比杆比洞（挂杆挂洞）：既比杆数差距，也比洞数胜负，有鸟鹰奖励
2. 比洞：只比洞数胜负，有鸟鹰奖励
3. 比杆：只比杆数差距，无鸟鹰奖励

核心规则：
- 支持2-4人参与，任意1v1配对
- 让杆系统：基于洞类型的个性化设置，打par/鸟/鹰不让杆
- 平洞处理：支持多种平洞算分规则和收取机制
- 鸟鹰奖励：基于原始杆数，打鸟洞分变成2分，打鹰洞分变成6分，HIO洞分变成11分
- 平洞收取：打鸟可收取2次平洞，打par可收取1次平洞，普通获胜不能收取
"""

logger = CustomLogger(name="DaoTest Stroke and Match Combo api", write_to_file=True)


# 数据模型定义
class Player(BaseModel):
    id: str
    name: str


class Pairing(BaseModel):
    player1_id: str
    player2_id: str
    # 让杆设置：player1对player2的让杆（正数表示让给player2）
    handicap_settings: Dict[str, float] = Field(default_factory=lambda: {
        "三杆洞": 0.0,
        "四杆洞": 0.0,
        "五杆洞": 0.0
    })


class GameConfig(BaseModel):
    mode: str = Field(..., description="游戏模式: 比杆比洞 | 比洞 | 比杆")

    # 平洞算分规则
    tie_scoring_rule: str = Field(default="平洞跳过(无肉)",
                                  description="平洞算分: 平洞跳过(无肉) | 平洞算1点 | 平洞算2点 | 平洞算3点 | 平洞算4点 | 平洞翻倍(不算鸟鹰奖) | 平洞翻倍(算鸟鹰奖) | 平洞连续翻番")

    # 让杆限制
    handicap_restrictions: Dict[str, bool] = Field(default_factory=lambda: {
        "par_bird_eagle_no_handicap": False  # 打par/鸟/鹰不让杆
    })


class PlayerScore(BaseModel):
    player_id: str
    raw_strokes: int


class Hole(BaseModel):
    hole_number: int
    par: int
    hole_type: str = Field(..., description="三杆洞 | 四杆洞 | 五杆洞")
    scores: List[PlayerScore]


class StrokeMatchGameData(BaseModel):
    game_config: GameConfig
    players: List[Player]
    pairings: List[Pairing]
    holes: List[Hole]


class StrokeMatchAPI:
    def __init__(self):
        self.logger = logger

    def calculate_stroke_match_score(self, game_data: StrokeMatchGameData) -> Dict[str, Any]:
        """比杆比洞计分主函数"""
        try:
            self.logger.info("=== 开始比杆比洞计分 ===")

            # 数据验证
            if not self._validate_game_data(game_data):
                return self._error_response("数据验证失败")

            # 初始化计分状态
            score_state = self._initialize_score_state(game_data)

            # 逐洞处理
            hole_details = []
            for hole in game_data.holes:
                hole_result = self._process_hole(hole, game_data, score_state)
                hole_details.append(hole_result)

                # 更新累积状态
                self._update_score_state(score_state, hole_result)

            # 最终结算
            final_summary = self._calculate_final_summary(game_data, score_state, hole_details)

            result = {
                "success": True,
                "data": {
                    "game_summary": final_summary,
                    "hole_details": hole_details
                },
                "message": "计分成功"
            }

            self.logger.info("=== 比杆比洞计分完成 ===")
            return result

        except Exception as e:
            self.logger.error(f"计分过程发生错误: {str(e)}")
            return self._error_response(f"计分失败: {str(e)}")

    def _validate_game_data(self, game_data: StrokeMatchGameData) -> bool:
        """验证游戏数据"""
        try:
            # 验证玩家数量
            if not (2 <= len(game_data.players) <= 4):
                self.logger.error(f"玩家人数错误: {len(game_data.players)}")
                return False

            # 验证配对
            if not game_data.pairings:
                self.logger.error("缺少配对信息")
                return False

            # 验证配对中的玩家ID是否存在
            player_ids = set(p.id for p in game_data.players)
            for pairing in game_data.pairings:
                if pairing.player1_id not in player_ids or pairing.player2_id not in player_ids:
                    self.logger.error(f"配对中的玩家ID不存在: {pairing.player1_id}, {pairing.player2_id}")
                    return False

            # 验证每个玩家至少参与一个配对
            involved_players = set()
            for pairing in game_data.pairings:
                involved_players.add(pairing.player1_id)
                involved_players.add(pairing.player2_id)

            if len(involved_players) < len(game_data.players):
                self.logger.error("存在未参与任何配对的玩家")
                return False

            # 验证洞次数据
            if not game_data.holes:
                self.logger.error("缺少洞次数据")
                return False

            for hole in game_data.holes:
                if len(hole.scores) != len(game_data.players):
                    self.logger.error(f"第{hole.hole_number}洞分数数据不完整")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"数据验证异常: {str(e)}")
            return False

    def _initialize_score_state(self, game_data: StrokeMatchGameData) -> Dict[str, Any]:
        """初始化计分状态"""
        # 为每个配对初始化状态
        pairing_states = {}
        for pairing in game_data.pairings:
            pairing_key = f"{pairing.player1_id}_vs_{pairing.player2_id}"
            pairing_states[pairing_key] = {
                "player1_total": 0,
                "player2_total": 0,
                "tie_count": 0,  # 累积的平洞次数
                "consecutive_ties": 0  # 连续平洞次数（用于连续翻番）
            }

        return {
            "pairing_states": pairing_states
        }

    def _process_hole(self, hole: Hole, game_data: StrokeMatchGameData, score_state: Dict[str, Any]) -> Dict[str, Any]:
        """处理单洞计分"""
        self.logger.info(f"开始处理第{hole.hole_number}洞")

        try:
            # 构建玩家分数映射
            player_scores = {score.player_id: score.raw_strokes for score in hole.scores}

            # 处理每个配对
            pairing_results = []
            for pairing in game_data.pairings:
                pairing_result = self._process_pairing(
                    hole, pairing, player_scores, game_data.game_config, score_state
                )
                pairing_results.append(pairing_result)

            # 计算特殊成就（鸟鹰等）
            special_achievements = self._calculate_special_achievements(hole, player_scores)

            return {
                "hole_number": hole.hole_number,
                "par": hole.par,
                "hole_type": hole.hole_type,
                "player_raw_scores": player_scores,
                "pairing_results": pairing_results,
                "special_achievements": special_achievements
            }

        except Exception as e:
            self.logger.error(f"处理第{hole.hole_number}洞时出错: {str(e)}")
            raise

    def _process_pairing(self, hole: Hole, pairing: Pairing, player_scores: Dict[str, int],
                         config: GameConfig, score_state: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个配对的计分"""
        pairing_key = f"{pairing.player1_id}_vs_{pairing.player2_id}"
        state = score_state["pairing_states"][pairing_key]

        # 获取原始杆数
        player1_raw = player_scores[pairing.player1_id]
        player2_raw = player_scores[pairing.player2_id]

        # 计算净杆数（应用让杆）
        player1_net, player2_net = self._calculate_net_scores(
            hole, pairing, player1_raw, player2_raw, config
        )

        # 计算基础分数
        stroke_diff = player2_net - player1_net  # 正数表示player1杆数少，获胜
        match_winner = 1 if player1_net < player2_net else (2 if player1_net > player2_net else 0)

        # 计算洞分（含鸟鹰奖励）
        hole_score_p1, hole_score_p2 = self._calculate_hole_scores(
            player1_raw, player2_raw, hole.par, match_winner, config
        )

        # 计算杆分
        stroke_score_p1, stroke_score_p2 = self._calculate_stroke_scores(
            stroke_diff, config
        )

        # 基础总分
        base_total_p1 = hole_score_p1 + stroke_score_p1
        base_total_p2 = hole_score_p2 + stroke_score_p2

        # 检查是否平洞
        is_tie = (base_total_p1 == base_total_p2)

        # 处理平洞逻辑
        tie_result = self._handle_tie_logic(
            is_tie, base_total_p1, base_total_p2, player1_raw, player2_raw,
            hole.par, state, config
        )

        return {
            "pairing": pairing_key,
            "player1_id": pairing.player1_id,
            "player2_id": pairing.player2_id,
            "player1_raw": player1_raw,
            "player2_raw": player2_raw,
            "player1_net": player1_net,
            "player2_net": player2_net,
            "base_scores": {
                "hole_score_p1": hole_score_p1,
                "hole_score_p2": hole_score_p2,
                "stroke_score_p1": stroke_score_p1,
                "stroke_score_p2": stroke_score_p2
            },
            "tie_info": tie_result["tie_info"],
            "final_scores": {
                "player1": tie_result["final_score_p1"],
                "player2": tie_result["final_score_p2"]
            }
        }

    def _calculate_net_scores(self, hole: Hole, pairing: Pairing, player1_raw: int,
                              player2_raw: int, config: GameConfig) -> tuple:
        """计算净杆数（应用让杆）"""
        # 获取让杆数
        handicap = pairing.handicap_settings.get(hole.hole_type, 0.0)

        # 应用让杆限制条件
        player1_handicap = self._apply_handicap_restrictions(
            player1_raw, 0, hole.par, config  # player1不被让杆
        )
        player2_handicap = self._apply_handicap_restrictions(
            player2_raw, handicap, hole.par, config  # player2被让杆
        )

        # 计算净杆数
        player1_net = player1_raw - player1_handicap
        player2_net = player2_raw - player2_handicap

        # 如果原始杆数超过par，让杆后不能比标准杆少
        if player2_raw > hole.par:
            player2_net = max(player2_net, hole.par)

        return player1_net, player2_net

    def _apply_handicap_restrictions(self, raw_score: int, handicap: float, par: int, config: GameConfig) -> float:
        """应用让杆限制条件"""
        restrictions = config.handicap_restrictions

        # 检查是否打出par/鸟/鹰（不让杆）
        if restrictions.get("par_bird_eagle_no_handicap", False):
            score_to_par = raw_score - par
            if score_to_par <= 0:  # Par或更好
                self.logger.info(f"选手打出{score_to_par}杆，不让杆")
                return 0.0

        return handicap

    def _calculate_hole_scores(self, player1_raw: int, player2_raw: int, par: int,
                               match_winner: int, config: GameConfig) -> tuple:
        """计算洞分（包含鸟鹰奖励）"""
        if config.mode == "比杆":
            return 0, 0

        hole_score_p1 = 0
        hole_score_p2 = 0

        if match_winner == 1:  # player1获胜
            # 特殊情况：三杆洞的HIO
            if par == 3 and player1_raw == 1:  # HIO
                hole_score_p1 = 11
            else:
                # 按正常鸟鹰规则
                score_to_par = player1_raw - par
                if score_to_par == -1:  # 打鸟
                    hole_score_p1 = 2
                elif score_to_par == -2:  # 打鹰
                    hole_score_p1 = 6
                elif score_to_par == -3:  # 信天翁
                    hole_score_p1 = 11
                else:
                    hole_score_p1 = 1  # 普通获胜1分

            hole_score_p2 = -hole_score_p1

        elif match_winner == 2:  # player2获胜
            # 特殊情况：三杆洞的HIO
            if par == 3 and player2_raw == 1:  # HIO
                hole_score_p2 = 11
            else:
                # 按正常鸟鹰规则
                score_to_par = player2_raw - par
                if score_to_par == -1:  # 打鸟
                    hole_score_p2 = 2
                elif score_to_par == -2:  # 打鹰
                    hole_score_p2 = 6
                elif score_to_par == -3:  # 信天翁
                    hole_score_p2 = 11
                else:
                    hole_score_p2 = 1  # 普通获胜1分

            hole_score_p1 = -hole_score_p2

        return hole_score_p1, hole_score_p2

    def _calculate_stroke_scores(self, stroke_diff: float, config: GameConfig) -> tuple:
        """计算杆分"""
        if config.mode == "比洞":
            return 0, 0

        # 杆分就是净杆数的差值
        return stroke_diff, -stroke_diff

    def _handle_tie_logic(self, is_tie: bool, base_total_p1: int, base_total_p2: int,
                          player1_raw: int, player2_raw: int, par: int,
                          state: Dict[str, Any], config: GameConfig) -> Dict[str, Any]:
        """处理平洞逻辑"""
        if is_tie:
            # 平洞处理
            tie_score = self._get_tie_score(config.tie_scoring_rule, state["consecutive_ties"])
            state["tie_count"] += tie_score
            state["consecutive_ties"] += 1

            return {
                "final_score_p1": 0,
                "final_score_p2": 0,
                "tie_info": {
                    "is_tie": True,
                    "tie_score": tie_score,
                    "accumulated_ties": state["tie_count"],
                    "consecutive_ties": state["consecutive_ties"]
                }
            }
        else:
            # 不是平洞，检查是否可以收取平洞分数
            winner = 1 if base_total_p1 > base_total_p2 else 2

            # 检查收取条件
            collected_ties = 0
            if winner == 1:
                collected_ties = self._check_tie_collection(player1_raw, par, state["tie_count"])
            else:
                collected_ties = self._check_tie_collection(player2_raw, par, state["tie_count"])

            # 扣除收取的平洞分数
            state["tie_count"] -= collected_ties
            state["consecutive_ties"] = 0  # 重置连续平洞

            # 计算最终分数
            final_p1 = base_total_p1 + (collected_ties if winner == 1 else -collected_ties)
            final_p2 = base_total_p2 + (collected_ties if winner == 2 else -collected_ties)

            return {
                "final_score_p1": final_p1,
                "final_score_p2": final_p2,
                "tie_info": {
                    "is_tie": False,
                    "collected_ties": collected_ties,
                    "remaining_ties": state["tie_count"],
                    "winner": winner
                }
            }

    def _get_tie_score(self, tie_rule: str, consecutive_ties: int) -> int:
        """获取平洞分数"""
        if tie_rule == "平洞跳过(无肉)":
            return 0
        elif tie_rule == "平洞算1点":
            return 1
        elif tie_rule == "平洞算2点":
            return 2
        elif tie_rule == "平洞算3点":
            return 3
        elif tie_rule == "平洞算4点":
            return 4
        elif tie_rule in ["平洞翻倍(不算鸟鹰奖)", "平洞翻倍(算鸟鹰奖)"]:
            return 2  # 基础翻倍
        elif tie_rule == "平洞连续翻番":
            return 2 ** consecutive_ties  # 指数翻倍
        else:
            return 0

    def _check_tie_collection(self, raw_score: int, par: int, available_ties: int) -> int:
        """检查平洞收取条件"""
        # 特殊情况：三杆洞的HIO
        if par == 3 and raw_score == 1:  # HIO
            return min(2, available_ties)

        score_to_par = raw_score - par

        if score_to_par == -1:  # 打鸟
            return min(2, available_ties)
        elif score_to_par == 0:  # 打par
            return min(1, available_ties)
        elif score_to_par <= -2:  # 打鹰或信天翁
            return min(2, available_ties)
        else:
            return 0  # 普通获胜不能收取

    def _calculate_special_achievements(self, hole: Hole, player_scores: Dict[str, int]) -> List[Dict[str, Any]]:
        """计算特殊成就"""
        achievements = []

        for player_id, raw_score in player_scores.items():
            score_to_par = raw_score - hole.par
            achievement_name = self._get_achievement_name(score_to_par, raw_score, hole.par)

            if score_to_par <= 0:  # 记录par及以上成就
                achievements.append({
                    "player_id": player_id,
                    "achievement": achievement_name,
                    "score_to_par": score_to_par,
                    "raw_score": raw_score
                })

        return achievements

    def _get_achievement_name(self, score_to_par: int, raw_score: int = None, par: int = None) -> str:
        """获取成就名称"""
        # 特殊情况：三杆洞的HIO
        if par is not None and raw_score is not None and par == 3 and raw_score == 1:
            return "一杆进洞"

        names = {
            -3: "信天翁",
            -2: "老鹰",
            -1: "小鸟",
            0: "标准杆",
            1: "柏忌",
            2: "双柏忌"
        }
        return names.get(score_to_par, f"+{score_to_par}杆")

    def _update_score_state(self, score_state: Dict[str, Any], hole_result: Dict[str, Any]):
        """更新累积计分状态"""
        for pairing_result in hole_result["pairing_results"]:
            pairing_key = pairing_result["pairing"]
            state = score_state["pairing_states"][pairing_key]

            # 更新总分
            state["player1_total"] += pairing_result["final_scores"]["player1"]
            state["player2_total"] += pairing_result["final_scores"]["player2"]

    def _calculate_final_summary(self, game_data: StrokeMatchGameData, score_state: Dict[str, Any],
                                 hole_details: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算最终总结"""
        # 汇总每个玩家的总分
        player_totals = {}
        for player in game_data.players:
            player_totals[player.id] = 0

        # 从所有配对中累加分数
        for pairing_key, state in score_state["pairing_states"].items():
            player1_id, player2_id = pairing_key.split("_vs_")
            player_totals[player1_id] += state["player1_total"]
            player_totals[player2_id] += state["player2_total"]

        # 构建最终分数列表
        final_scores = []
        for player in game_data.players:
            final_scores.append({
                "player_id": player.id,
                "player_name": player.name,
                "total_score": player_totals[player.id]
            })

        # 构建配对总结
        pairing_summaries = []
        for pairing in game_data.pairings:
            pairing_key = f"{pairing.player1_id}_vs_{pairing.player2_id}"
            state = score_state["pairing_states"][pairing_key]

            pairing_summaries.append({
                "pairing": pairing_key,
                "player1_id": pairing.player1_id,
                "player2_id": pairing.player2_id,
                "player1_total": state["player1_total"],
                "player2_total": state["player2_total"],
                "remaining_ties": state["tie_count"]
            })

        return {
            "total_holes": len(game_data.holes),
            "game_mode": game_data.game_config.mode,
            "tie_scoring_rule": game_data.game_config.tie_scoring_rule,
            "players": [{"id": p.id, "name": p.name} for p in game_data.players],
            "pairings": pairing_summaries,
            "final_scores": final_scores
        }

    def _error_response(self, message: str) -> Dict[str, Any]:
        """返回错误响应"""
        return {
            "success": False,
            "error_code": "CALCULATION_ERROR",
            "message": message,
            "data": None
        }


# 接口函数
def calculate_stroke_match_score(game_data: StrokeMatchGameData) -> Dict[str, Any]:
    """比杆比洞计分接口函数"""
    logger.info("------------------比杆比洞计分开始--------------------")

    try:
        api = StrokeMatchAPI()
        result = api.calculate_stroke_match_score(game_data)

        if result["success"]:
            logger.info("比杆比洞计分成功完成")
        else:
            logger.error(f"比杆比洞计分失败: {result['message']}")

        return result

    except Exception as e:
        logger.error(f"比杆比洞计分接口异常: {str(e)}")
        return {
            "success": False,
            "error_code": "INTERFACE_ERROR",
            "message": f"接口调用异常: {str(e)}",
            "data": None
        }
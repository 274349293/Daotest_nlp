import json
import copy
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from utils.nlp_logging import CustomLogger

logger = CustomLogger(name="DaoTest Stroke and Match Combo api", write_to_file=True)


# ===== 数据模型 =====
class Player(BaseModel):
    id: str
    name: str


class Pairing(BaseModel):
    player1_id: str
    player2_id: str
    handicap_settings: Dict[str, float] = Field(default_factory=lambda: {
        "三杆洞": 0.0,
        "四杆洞": 0.0,
        "五杆洞": 0.0,
    })


class GameConfig(BaseModel):
    mode: str = Field(..., description="游戏模式: 比杆比洞 | 比洞 | 比杆")
    # 新增：将 pairings 移入 game_config
    pairings: List[Pairing] = Field(default_factory=list, description="配对信息列表")

    tie_scoring_rule: str = Field(
        default="平洞跳过(无肉)",
        description=(
            "平洞算分: 平洞跳过(无肉) | 平洞算1点 | 平洞算2点 | 平洞算3点 | 平洞算4点 | "
            "平洞翻倍(不算鸟鹰奖) | 平洞翻倍(算鸟鹰奖) | 平洞连续翻番"
        ),
    )
    handicap_restrictions: Dict[str, bool] = Field(
        default_factory=lambda: {"par_bird_eagle_no_handicap": False}
    )


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
    holes: List[Hole]

    # 兼容旧版：接受顶层 pairings（不推荐）
    pairings_legacy: Optional[List[Pairing]] = Field(None, alias="pairings")


# ===== 业务实现 =====
class StrokeMatchAPI:
    def __init__(self):
        self.logger = logger

    # ---- 工具：统一获取 pairings（新 > 旧） ----
    def _get_pairings(self, game_data: StrokeMatchGameData) -> List[Pairing]:
        if game_data.game_config.pairings:
            return game_data.game_config.pairings
        return game_data.pairings_legacy or []

    def calculate_stroke_match_score(self, game_data: StrokeMatchGameData) -> Dict[str, Any]:
        try:
            self.logger.info("=== 开始比杆比洞计分 ===")

            if not self._validate_game_data(game_data):
                return self._error_response("数据验证失败")

            score_state = self._initialize_score_state(game_data)
            hole_details = []
            for hole in game_data.holes:
                hole_result = self._process_hole(hole, game_data, score_state)
                hole_details.append(hole_result)
                self._update_score_state(score_state, hole_result)

            final_summary = self._calculate_final_summary(game_data, score_state, hole_details)
            result = {"success": True, "data": {"game_summary": final_summary, "hole_details": hole_details},
                      "message": "计分成功"}
            self.logger.info("=== 比杆比洞计分完成 ===")
            return result
        except Exception as e:
            self.logger.error(f"计分过程发生错误: {str(e)}")
            return self._error_response(f"计分失败: {str(e)}")

    def _validate_game_data(self, game_data: StrokeMatchGameData) -> bool:
        try:
            if not (2 <= len(game_data.players) <= 4):
                self.logger.error(f"玩家人数错误: {len(game_data.players)}")
                return False

            pairings = self._get_pairings(game_data)
            if not pairings:
                self.logger.error("缺少配对信息")
                return False

            player_ids = {p.id for p in game_data.players}
            for pairing in pairings:
                if pairing.player1_id not in player_ids or pairing.player2_id not in player_ids:
                    self.logger.error(f"配对中的玩家ID不存在: {pairing.player1_id}, {pairing.player2_id}")
                    return False

            involved = {pid for pr in pairings for pid in (pr.player1_id, pr.player2_id)}
            if len(involved) < len(game_data.players):
                self.logger.error("存在未参与任何配对的玩家")
                return False

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
        pairing_states: Dict[str, Dict[str, Any]] = {}
        for pairing in self._get_pairings(game_data):
            key = f"{pairing.player1_id}_vs_{pairing.player2_id}"
            pairing_states[key] = {"player1_total": 0, "player2_total": 0, "tie_count": 0, "consecutive_ties": 0}
        return {"pairing_states": pairing_states}

    def _process_hole(self, hole: Hole, game_data: StrokeMatchGameData, score_state: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"开始处理第{hole.hole_number}洞")
        player_scores = {s.player_id: s.raw_strokes for s in hole.scores}

        pairing_results = []
        for pairing in self._get_pairings(game_data):
            pairing_results.append(
                self._process_pairing(hole, pairing, player_scores, game_data.game_config, score_state)
            )

        special_achievements = self._calculate_special_achievements(hole, player_scores)
        return {
            "hole_number": hole.hole_number,
            "par": hole.par,
            "hole_type": hole.hole_type,
            "player_raw_scores": player_scores,
            "pairing_results": pairing_results,
            "special_achievements": special_achievements,
        }

    def _process_pairing(self, hole: Hole, pairing: Pairing, player_scores: Dict[str, int], config: GameConfig,
                         score_state: Dict[str, Any]) -> Dict[str, Any]:
        key = f"{pairing.player1_id}_vs_{pairing.player2_id}"
        state = score_state["pairing_states"][key]

        p1_raw = player_scores[pairing.player1_id]
        p2_raw = player_scores[pairing.player2_id]
        p1_net, p2_net = self._calculate_net_scores(hole, pairing, p1_raw, p2_raw, config)

        stroke_diff = p2_net - p1_net
        match_winner = 1 if p1_net < p2_net else (2 if p1_net > p2_net else 0)

        hole_score_p1, hole_score_p2 = self._calculate_hole_scores(p1_raw, p2_raw, hole.par, match_winner, config)
        stroke_score_p1, stroke_score_p2 = self._calculate_stroke_scores(stroke_diff, config)

        base_p1 = hole_score_p1 + stroke_score_p1
        base_p2 = hole_score_p2 + stroke_score_p2
        is_tie = base_p1 == base_p2

        tie_result = self._handle_tie_logic(is_tie, base_p1, base_p2, p1_raw, p2_raw, hole.par, state, config)
        return {
            "pairing": key,
            "player1_id": pairing.player1_id,
            "player2_id": pairing.player2_id,
            "player1_raw": p1_raw,
            "player2_raw": p2_raw,
            "player1_net": p1_net,
            "player2_net": p2_net,
            "base_scores": {
                "hole_score_p1": hole_score_p1,
                "hole_score_p2": hole_score_p2,
                "stroke_score_p1": stroke_score_p1,
                "stroke_score_p2": stroke_score_p2,
            },
            "tie_info": tie_result["tie_info"],
            "final_scores": {"player1": tie_result["final_score_p1"], "player2": tie_result["final_score_p2"]},
        }

    def _apply_handicap_restrictions(self, raw_score: int, handicap: float, par: int, config: GameConfig) -> float:
        if config.handicap_restrictions.get("par_bird_eagle_no_handicap", False):
            if raw_score - par <= 0:
                self.logger.info("选手打出≤Par，不让杆")
                return 0.0
        return handicap

    def _calculate_net_scores(self, hole: Hole, pairing: Pairing, p1_raw: int, p2_raw: int, config: GameConfig) -> \
    Tuple[float, float]:
        handicap = pairing.handicap_settings.get(hole.hole_type, 0.0)
        p1_h = self._apply_handicap_restrictions(p1_raw, 0, hole.par, config)
        p2_h = self._apply_handicap_restrictions(p2_raw, handicap, hole.par, config)

        p1_net = p1_raw - p1_h
        p2_net = p2_raw - p2_h
        if p2_raw > hole.par:
            p2_net = max(p2_net, hole.par)
        return p1_net, p2_net

    def _calculate_hole_scores(self, p1_raw: int, p2_raw: int, par: int, match_winner: int, config: GameConfig) -> \
    Tuple[int, int]:
        if config.mode == "比杆":
            return 0, 0
        hs1 = hs2 = 0
        if match_winner == 1:
            if par == 3 and p1_raw == 1:
                hs1 = 11
            else:
                d = p1_raw - par
                hs1 = 2 if d == -1 else 6 if d == -2 else 11 if d == -3 else 1
            hs2 = -hs1
        elif match_winner == 2:
            if par == 3 and p2_raw == 1:
                hs2 = 11
            else:
                d = p2_raw - par
                hs2 = 2 if d == -1 else 6 if d == -2 else 11 if d == -3 else 1
            hs1 = -hs2
        return hs1, hs2

    def _calculate_stroke_scores(self, stroke_diff: float, config: GameConfig) -> Tuple[float, float]:
        if config.mode == "比洞":
            return 0, 0
        return stroke_diff, -stroke_diff

    def _get_tie_score(self, tie_rule: str, consecutive_ties: int) -> int:
        if tie_rule == "平洞跳过(无肉)":
            return 0
        if tie_rule == "平洞算1点":
            return 1
        if tie_rule == "平洞算2点":
            return 2
        if tie_rule == "平洞算3点":
            return 3
        if tie_rule == "平洞算4点":
            return 4
        if tie_rule in ["平洞翻倍(不算鸟鹰奖)", "平洞翻倍(算鸟鹰奖)"]:
            return 2
        if tie_rule == "平洞连续翻番":
            return 2 ** consecutive_ties
        return 0

    def _check_tie_collection(self, raw_score: int, par: int, available_ties: int) -> int:
        if par == 3 and raw_score == 1:
            return min(2, available_ties)
        d = raw_score - par
        if d <= -2:
            return min(2, available_ties)
        if d == -1:
            return min(2, available_ties)
        if d == 0:
            return min(1, available_ties)
        return 0

    def _handle_tie_logic(self, is_tie: bool, base_p1: int, base_p2: int, p1_raw: int, p2_raw: int, par: int,
                          state: Dict[str, Any], config: GameConfig) -> Dict[str, Any]:
        if is_tie:
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
                    "consecutive_ties": state["consecutive_ties"],
                },
            }
        winner = 1 if base_p1 > base_p2 else 2
        collected = self._check_tie_collection(p1_raw if winner == 1 else p2_raw, par, state["tie_count"])
        state["tie_count"] -= collected
        state["consecutive_ties"] = 0
        return {
            "final_score_p1": base_p1 + (collected if winner == 1 else -collected),
            "final_score_p2": base_p2 + (collected if winner == 2 else -collected),
            "tie_info": {"is_tie": False, "collected_ties": collected, "remaining_ties": state["tie_count"],
                         "winner": winner},
        }

    def _calculate_special_achievements(self, hole: Hole, player_scores: Dict[str, int]) -> List[Dict[str, Any]]:
        out = []
        for pid, raw in player_scores.items():
            d = raw - hole.par
            name = self._get_achievement_name(d, raw, hole.par)
            if d <= 0:
                out.append({"player_id": pid, "achievement": name, "score_to_par": d, "raw_score": raw})
        return out

    def _get_achievement_name(self, d: int, raw: Optional[int] = None, par: Optional[int] = None) -> str:
        if par is not None and raw is not None and par == 3 and raw == 1:
            return "一杆进洞"
        mapping = {-3: "信天翁", -2: "老鹰", -1: "小鸟", 0: "标准杆", 1: "柏忌", 2: "双柏忌"}
        return mapping.get(d, f"+{d}杆")

    def _update_score_state(self, score_state: Dict[str, Any], hole_result: Dict[str, Any]):
        for pr in hole_result["pairing_results"]:
            key = pr["pairing"]
            state = score_state["pairing_states"][key]
            state["player1_total"] += pr["final_scores"]["player1"]
            state["player2_total"] += pr["final_scores"]["player2"]

    def _calculate_final_summary(self, game_data: StrokeMatchGameData, score_state: Dict[str, Any],
                                 hole_details: List[Dict[str, Any]]) -> Dict[str, Any]:
        player_totals: Dict[str, float] = {p.id: 0 for p in game_data.players}
        for key, st in score_state["pairing_states"].items():
            p1, p2 = key.split("_vs_")
            player_totals[p1] += st["player1_total"]
            player_totals[p2] += st["player2_total"]

        final_scores = [{"player_id": p.id, "player_name": p.name, "total_score": player_totals[p.id]} for p in
                        game_data.players]

        pairing_summaries = []
        for pairing in self._get_pairings(game_data):
            key = f"{pairing.player1_id}_vs_{pairing.player2_id}"
            st = score_state["pairing_states"][key]
            pairing_summaries.append({
                "pairing": key,
                "player1_id": pairing.player1_id,
                "player2_id": pairing.player2_id,
                "player1_total": st["player1_total"],
                "player2_total": st["player2_total"],
                "remaining_ties": st["tie_count"],
            })

        return {
            "total_holes": len(game_data.holes),
            "game_mode": game_data.game_config.mode,
            "tie_scoring_rule": game_data.game_config.tie_scoring_rule,
            "players": [{"id": p.id, "name": p.name} for p in game_data.players],
            "pairings": pairing_summaries,
            "final_scores": final_scores,
        }

    def _error_response(self, message: str) -> Dict[str, Any]:
        return {"success": False, "error_code": "CALCULATION_ERROR", "message": message, "data": None}


# 对外函数
def calculate_stroke_match_score(game_data: StrokeMatchGameData) -> Dict[str, Any]:
    logger.info("------------------比杆比洞计分开始--------------------")
    try:
        api = StrokeMatchAPI()
        result = api.calculate_stroke_match_score(game_data)
        if result.get("success"):
            logger.info("比杆比洞计分成功完成")
        else:
            logger.error(f"比杆比洞计分失败: {result.get('message')}")
        return result
    except Exception as e:
        logger.error(f"比杆比洞计分接口异常: {str(e)}")
        return {"success": False, "error_code": "INTERFACE_ERROR", "message": f"接口调用异常: {str(e)}", "data": None}

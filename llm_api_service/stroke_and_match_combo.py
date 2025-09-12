import json
import copy
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from utils.nlp_logging import CustomLogger

# 创建logger但避免重复输出
logger = CustomLogger(name="StrokeMatch Updated API", write_to_file=True)


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
    pairings: List[Pairing] = Field(default_factory=list, description="配对信息列表")

    # 简化的配置项
    tie_scoring_rule: str = Field(
        default="平洞算1点",
        description="平洞算分: 平洞算1点 | 平洞翻倍(不算鸟鹰奖)"
    )

    # 让杆限制 - 简化为只有par/鸟/鹰不让杆的选项
    handicap_restrictions: Dict[str, bool] = Field(
        default_factory=lambda: {"par_bird_eagle_no_handicap": False}
    )

    # 修改：改为bird_eagle_reward字段，支持两种格式
    bird_eagle_reward: str = Field(
        default="鸟+1/鹰+5/HIO+10",
        description="鸟鹰奖励算分: 鸟+1/鹰+5/HIO+10 | 鸟*2/鹰*5/HIO*10"
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


# ===== 业务实现 =====
class UpdatedStrokeMatchAPI:
    def __init__(self):
        self.logger = logger
        # 乘法模式的倍数
        self.bird_eagle_multipliers = {
            -1: 2,  # 小鸟 *2
            -2: 5,  # 老鹰 *5
            -3: 10,  # 信天翁/一杆进洞 *10
        }
        # 加法模式的奖励分数
        self.bird_eagle_bonus_points = {
            -1: 1,  # 小鸟 +1
            -2: 5,  # 老鹰 +5
            -3: 10,  # 信天翁/一杆进洞 +10
        }

    def _get_bird_eagle_mode_and_values(self, bird_eagle_reward: str) -> Tuple[str, Dict[int, int]]:
        """
        解析bird_eagle_reward参数，返回模式和对应的数值字典
        """
        if "*" in bird_eagle_reward:
            # 乘法模式：鸟*2/鹰*5/HIO*10
            return "乘法模式", self.bird_eagle_multipliers
        else:
            # 加法模式：鸟+1/鹰+5/HIO+10
            return "加法模式", self.bird_eagle_bonus_points

    def calculate_stroke_match_score(self, game_data: StrokeMatchGameData) -> Dict[str, Any]:
        try:
            self.logger.info("=" * 80)
            self.logger.info("开始挂杆挂洞计分")
            self.logger.info("=" * 80)

            # 保存数据供日志使用
            self.game_data = game_data

            if not self._validate_game_data(game_data):
                return self._error_response("数据验证失败")

            score_state = self._initialize_score_state(game_data)
            hole_details = []

            for hole in game_data.holes:
                hole_result = self._process_hole(hole, game_data, score_state)
                hole_details.append(hole_result)
                self._update_score_state(score_state, hole_result)

            final_summary = self._calculate_final_summary(game_data, score_state, hole_details)
            result = {
                "success": True,
                "data": {
                    "game_summary": final_summary,
                    "hole_details": hole_details
                },
                "message": "计分成功"
            }
            self.logger.info("=" * 80)
            self.logger.info("挂杆挂洞计分完成")
            self.logger.info("=" * 80)
            return result

        except Exception as e:
            self.logger.error(f"计分过程发生错误: {str(e)}")
            return self._error_response(f"计分失败: {str(e)}")

    def _validate_game_data(self, game_data: StrokeMatchGameData) -> bool:
        try:
            if not (2 <= len(game_data.players) <= 4):
                self.logger.error(f"玩家人数错误: {len(game_data.players)}")
                return False

            if not game_data.game_config.pairings:
                self.logger.error("缺少配对信息")
                return False

            player_ids = {p.id for p in game_data.players}
            for pairing in game_data.game_config.pairings:
                if pairing.player1_id not in player_ids or pairing.player2_id not in player_ids:
                    self.logger.error(f"配对中的玩家ID不存在: {pairing.player1_id}, {pairing.player2_id}")
                    return False

            if not game_data.holes:
                self.logger.error("缺少洞次数据")
                return False

            for hole in game_data.holes:
                if len(hole.scores) != len(game_data.players):
                    self.logger.error(f"第{hole.hole_number}洞分数数据不完整")
                    return False

            # 验证鸟鹰奖励参数格式
            if not (("+" in game_data.game_config.bird_eagle_reward) or (
                    "*" in game_data.game_config.bird_eagle_reward)):
                self.logger.error(f"鸟鹰奖励参数格式无效: {game_data.game_config.bird_eagle_reward}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"数据验证异常: {str(e)}")
            return False

    def _initialize_score_state(self, game_data: StrokeMatchGameData) -> Dict[str, Any]:
        """初始化计分状态"""
        pairing_states: Dict[str, Dict[str, Any]] = {}
        for pairing in game_data.game_config.pairings:
            key = f"{pairing.player1_id}_vs_{pairing.player2_id}"
            pairing_states[key] = {
                "player1_total": 0,
                "player2_total": 0,
                "tie_count": 0,
                "tie_accumulated_score": 0
            }
        return {"pairing_states": pairing_states}

    def _process_hole(self, hole: Hole, game_data: StrokeMatchGameData, score_state: Dict[str, Any]) -> Dict[str, Any]:
        """处理单洞计分"""
        self.logger.info(f"\n{'=' * 80}")
        self.logger.info(f"第{hole.hole_number}洞 - Par {hole.par} ({hole.hole_type})")
        self.logger.info(f"{'=' * 80}")

        player_scores = {s.player_id: s.raw_strokes for s in hole.scores}

        # 显示原始杆数
        self.logger.info(f"原始杆数:")
        for player in game_data.players:
            player_name = player.name
            raw_strokes = player_scores[player.id]
            to_par = raw_strokes - hole.par
            to_par_str = f"{to_par:+d}" if to_par != 0 else "E"
            achievement = self._get_achievement_name(to_par, raw_strokes, hole.par)
            self.logger.info(f"  {player_name}: {raw_strokes}杆 ({to_par_str}) - {achievement}")

        pairing_results = []
        for pairing in game_data.game_config.pairings:
            pairing_results.append(
                self._process_pairing(hole, pairing, player_scores, game_data.game_config, score_state)
            )

        special_achievements = self._calculate_special_achievements(hole, player_scores)

        # 显示本洞汇总
        self.logger.info(f"\n{'-' * 60}")
        self.logger.info(f"第{hole.hole_number}洞分数汇总:")
        for pr in pairing_results:
            p1_name = next(p.name for p in game_data.players if p.id == pr["player1_id"])
            p2_name = next(p.name for p in game_data.players if p.id == pr["player2_id"])
            p1_score = pr["final_scores"]["player1"]
            p2_score = pr["final_scores"]["player2"]
            self.logger.info(f"  {p1_name}: {p1_score:+.1f}分")
            self.logger.info(f"  {p2_name}: {p2_score:+.1f}分")
        self.logger.info(f"{'-' * 60}")

        return {
            "hole_number": hole.hole_number,
            "par": hole.par,
            "hole_type": hole.hole_type,
            "player_raw_scores": player_scores,
            "pairing_results": pairing_results,
            "special_achievements": special_achievements,
        }

    def _process_pairing(self, hole: Hole, pairing: Pairing, player_scores: Dict[str, int],
                         config: GameConfig, score_state: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个配对的计分"""
        key = f"{pairing.player1_id}_vs_{pairing.player2_id}"
        state = score_state["pairing_states"][key]

        # 获取选手信息
        p1_name = next(p.name for p in self.game_data.players if p.id == pairing.player1_id)
        p2_name = next(p.name for p in self.game_data.players if p.id == pairing.player2_id)

        p1_raw = player_scores[pairing.player1_id]
        p2_raw = player_scores[pairing.player2_id]

        self.logger.info(f"\n配对: {p1_name} vs {p2_name}")
        self.logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # 计算净杆数（应用让杆）
        p1_net, p2_net = self._calculate_net_scores(hole, pairing, p1_raw, p2_raw, config, p1_name, p2_name)

        # 计算杆分差距
        stroke_diff = p1_net - p2_net  # 正数表示p1比p2多杆，负数表示p1比p2少杆

        # 确定洞数胜负
        match_winner = 0  # 0=平局, 1=p1胜, 2=p2胜
        winner_name = ""
        if p1_net < p2_net:
            match_winner = 1
            winner_name = p1_name
        elif p1_net > p2_net:
            match_winner = 2
            winner_name = p2_name

        self.logger.info(f"\n胜负判定:")
        if match_winner == 0:
            self.logger.info(f"   洞数: 平局 ({p1_net} vs {p2_net})")
        else:
            net_diff = abs(p1_net - p2_net)
            self.logger.info(f"   洞数: {winner_name}胜{net_diff}杆 ({p1_net} vs {p2_net})")

        # 计算基础分数
        hole_score_p1, hole_score_p2 = self._calculate_hole_scores(p1_raw, p2_raw, hole.par, match_winner, config)
        stroke_score_p1, stroke_score_p2 = self._calculate_stroke_scores(stroke_diff, config)

        self.logger.info(f"\n基础分数计算:")
        if config.mode in ["比杆比洞", "比洞"]:
            self.logger.info(f"   洞分: {p1_name} {hole_score_p1:+.1f}分, {p2_name} {hole_score_p2:+.1f}分")
        if config.mode in ["比杆比洞", "比杆"]:
            stroke_diff_display = stroke_diff if stroke_diff != 0 else 0
            self.logger.info(
                f"   杆分: {p1_name} {stroke_score_p1:+.1f}分, {p2_name} {stroke_score_p2:+.1f}分 (杆数差: {stroke_diff_display:+.1f})")

        # 基础总分
        base_score_p1 = hole_score_p1 + stroke_score_p1
        base_score_p2 = hole_score_p2 + stroke_score_p2

        self.logger.info(f"   基础总分: {p1_name} {base_score_p1:+.1f}分, {p2_name} {base_score_p2:+.1f}分")

        # 应用鸟鹰奖励（只影响洞分部分）
        bird_eagle_bonus_p1, bird_eagle_bonus_p2 = self._apply_bird_eagle_bonus(
            p1_raw, p2_raw, hole.par, match_winner, hole_score_p1, hole_score_p2, p1_name, p2_name, config
        )

        # 计算最终基础分（含鸟鹰奖励）
        final_base_p1 = stroke_score_p1 + bird_eagle_bonus_p1  # 杆分 + 鸟鹰洞分
        final_base_p2 = stroke_score_p2 + bird_eagle_bonus_p2

        if bird_eagle_bonus_p1 != hole_score_p1 or bird_eagle_bonus_p2 != hole_score_p2:
            self.logger.info(f"   鸟鹰奖励后: {p1_name} {final_base_p1:+.1f}分, {p2_name} {final_base_p2:+.1f}分")

        # 处理平洞逻辑
        tie_result = self._handle_tie_logic(final_base_p1, final_base_p2, p1_raw, p2_raw, hole.par, state, config,
                                            p1_name, p2_name)

        self.logger.info(f"\n最终本洞分数:")
        self.logger.info(f"   {p1_name}: {tie_result['final_score_p1']:+.1f}分")
        self.logger.info(f"   {p2_name}: {tie_result['final_score_p2']:+.1f}分")

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
                "bird_eagle_bonus_p1": bird_eagle_bonus_p1 - hole_score_p1,  # 只记录额外奖励部分
                "bird_eagle_bonus_p2": bird_eagle_bonus_p2 - hole_score_p2,
            },
            "tie_info": tie_result["tie_info"],
            "final_scores": {
                "player1": tie_result["final_score_p1"],
                "player2": tie_result["final_score_p2"]
            },
        }

    def _calculate_net_scores(self, hole: Hole, pairing: Pairing, p1_raw: int, p2_raw: int,
                              config: GameConfig, p1_name: str, p2_name: str) -> Tuple[float, float]:
        """计算净杆数（应用让杆）"""
        # 获取让杆数 - p1对p2的让杆
        handicap = pairing.handicap_settings.get(hole.hole_type, 0.0)

        self.logger.info(f"\n让杆计算:")
        if handicap > 0:
            self.logger.info(f"   {p1_name}让{p2_name} {handicap}杆 ({hole.hole_type})")
        else:
            self.logger.info(f"   无让杆")

        # 检查让杆限制
        p1_handicap = self._apply_handicap_restrictions(p1_raw, 0, hole.par, config, p1_name)
        p2_handicap = self._apply_handicap_restrictions(p2_raw, handicap, hole.par, config, p2_name)

        # 计算净杆数
        p1_net = p1_raw - p1_handicap
        p2_net = p2_raw - p2_handicap

        # 重要：被让杆者净杆数不能低于标准杆
        original_p2_net = p2_net
        if p2_raw > hole.par:
            p2_net = max(p2_net, hole.par)
            if original_p2_net != p2_net:
                self.logger.info(f"   {p2_name}净杆数调整: {original_p2_net} -> {p2_net} (不能低于Par{hole.par})")

        self.logger.info(f"   净杆数: {p1_name} {p1_net}杆, {p2_name} {p2_net}杆")

        return p1_net, p2_net

    def _apply_handicap_restrictions(self, raw_score: int, handicap: float, par: int, config: GameConfig,
                                     player_name: str) -> float:
        """应用让杆限制条件"""
        if config.handicap_restrictions.get("par_bird_eagle_no_handicap", False):
            if raw_score <= par:  # 打出Par或更好成绩
                if handicap > 0:
                    self.logger.info(f"   {player_name}打出Par/鸟/鹰不让杆 (原让杆: {handicap})")
                return 0.0
        return handicap

    def _calculate_hole_scores(self, p1_raw: int, p2_raw: int, par: int, match_winner: int, config: GameConfig) -> \
            Tuple[int, int]:
        """计算洞分（胜负分）"""
        if config.mode == "比杆":
            return 0, 0  # 比杆模式不算洞分

        if match_winner == 1:
            return 1, -1  # p1胜
        elif match_winner == 2:
            return -1, 1  # p2胜
        else:
            return 0, 0  # 平局

    def _calculate_stroke_scores(self, stroke_diff: float, config: GameConfig) -> Tuple[float, float]:
        """计算杆分"""
        if config.mode == "比洞":
            return 0, 0  # 比洞模式不算杆分

        # stroke_diff = p1_net - p2_net
        # 正数表示p1比p2多杆，所以p1得负分，p2得正分
        return -stroke_diff, stroke_diff

    def _apply_bird_eagle_bonus(self, p1_raw: int, p2_raw: int, par: int, match_winner: int,
                                hole_score_p1: int, hole_score_p2: int, p1_name: str, p2_name: str,
                                config: GameConfig) -> Tuple[float, float]:
        """应用鸟鹰奖励，支持乘法模式和加法模式"""
        self.logger.info(f"\n鸟鹰奖励检查:")

        # 解析奖励模式和对应数值
        reward_mode, reward_values = self._get_bird_eagle_mode_and_values(config.bird_eagle_reward)
        self.logger.info(f"   奖励模式: {reward_mode} ({config.bird_eagle_reward})")

        # 只有获胜者且打出鸟鹰才能获得奖励
        if match_winner == 1:
            p1_to_par = p1_raw - par
            if p1_to_par <= -1 and p1_to_par >= -3:  # 小鸟到信天翁
                achievement = self._get_achievement_name(p1_to_par, p1_raw, par)

                if reward_mode == "乘法模式":
                    # 乘法模式：洞分 × 倍数
                    multiplier = reward_values[p1_to_par]
                    new_p1_hole_score = hole_score_p1 * multiplier
                    new_p2_hole_score = -new_p1_hole_score  # 保持零和

                    self.logger.info(
                        f"   {p1_name}打出{achievement}且获胜，洞分 {hole_score_p1} × {multiplier} = {new_p1_hole_score}")
                    self.logger.info(f"   {p2_name}洞分相应调整为 {new_p2_hole_score} (保持零和)")

                else:  # 加法模式
                    # 加法模式：洞分 + 奖励分
                    bonus_points = reward_values[p1_to_par]
                    new_p1_hole_score = hole_score_p1 + bonus_points
                    new_p2_hole_score = hole_score_p2 - bonus_points  # 保持零和

                    self.logger.info(
                        f"   {p1_name}打出{achievement}且获胜，洞分 {hole_score_p1} + {bonus_points} = {new_p1_hole_score}")
                    self.logger.info(f"   {p2_name}洞分相应调整为 {new_p2_hole_score} (保持零和)")

                return new_p1_hole_score, new_p2_hole_score
            else:
                self.logger.info(f"   {p1_name}获胜但未打出鸟鹰，无额外奖励")

        elif match_winner == 2:
            p2_to_par = p2_raw - par
            if p2_to_par <= -1 and p2_to_par >= -3:  # 小鸟到信天翁
                achievement = self._get_achievement_name(p2_to_par, p2_raw, par)

                if reward_mode == "乘法模式":
                    # 乘法模式：洞分 × 倍数
                    multiplier = reward_values[p2_to_par]
                    new_p2_hole_score = hole_score_p2 * multiplier
                    new_p1_hole_score = -new_p2_hole_score  # 保持零和

                    self.logger.info(
                        f"   {p2_name}打出{achievement}且获胜，洞分 {hole_score_p2} × {multiplier} = {new_p2_hole_score}")
                    self.logger.info(f"   {p1_name}洞分相应调整为 {new_p1_hole_score} (保持零和)")

                else:  # 加法模式
                    # 加法模式：洞分 + 奖励分
                    bonus_points = reward_values[p2_to_par]
                    new_p2_hole_score = hole_score_p2 + bonus_points
                    new_p1_hole_score = hole_score_p1 - bonus_points  # 保持零和

                    self.logger.info(
                        f"   {p2_name}打出{achievement}且获胜，洞分 {hole_score_p2} + {bonus_points} = {new_p2_hole_score}")
                    self.logger.info(f"   {p1_name}洞分相应调整为 {new_p1_hole_score} (保持零和)")

                return new_p1_hole_score, new_p2_hole_score
            else:
                self.logger.info(f"   {p2_name}获胜但未打出鸟鹰，无额外奖励")
        else:
            self.logger.info(f"   平局，无鸟鹰奖励")

        return hole_score_p1, hole_score_p2

    def _handle_tie_logic(self, base_p1: float, base_p2: float, p1_raw: int, p2_raw: int, par: int,
                          state: Dict[str, Any], config: GameConfig, p1_name: str, p2_name: str) -> Dict[str, Any]:
        """处理平洞逻辑"""
        is_tie = (base_p1 == base_p2)

        self.logger.info(f"\n平洞逻辑处理:")
        self.logger.info(f"   本洞得分: {p1_name} {base_p1:+.1f}分 vs {p2_name} {base_p2:+.1f}分")

        if is_tie:
            # 平洞处理
            tie_score = self._get_tie_score(config.tie_scoring_rule)
            if config.tie_scoring_rule == "平洞翻倍(不算鸟鹰奖)":
                state["tie_count"] += 1
                self.logger.info(f"   本洞平洞，累积平洞次数+1，当前共{state['tie_count']}次")
            else:
                state["tie_accumulated_score"] += tie_score
                state["tie_count"] += 1
                self.logger.info(
                    f"   本洞平洞，累积{tie_score}分，当前共{state['tie_accumulated_score']}分({state['tie_count']}次)")

            return {
                "final_score_p1": 0,
                "final_score_p2": 0,
                "tie_info": {
                    "is_tie": True,
                    "tie_score": tie_score,
                    "accumulated_ties": state["tie_count"],
                    "accumulated_score": state["tie_accumulated_score"],
                },
            }
        else:
            # 不是平洞，检查是否收取累积分数
            collected_score = 0
            if state["tie_count"] > 0:
                self.logger.info(
                    f"   非平洞，当前累积平洞: {state['tie_count']}次/{state.get('tie_accumulated_score', 0)}分")
                collected_score = self._collect_tie_score(base_p1, base_p2, p1_raw, p2_raw, par, state, config, p1_name,
                                                          p2_name)
            else:
                self.logger.info(f"   非平洞，无累积平洞分数")

            winner_name = p1_name if base_p1 > base_p2 else p2_name

            final_p1 = base_p1 + collected_score if base_p1 > base_p2 else base_p1 - collected_score
            final_p2 = base_p2 - collected_score if base_p1 > base_p2 else base_p2 + collected_score

            if collected_score > 0:
                self.logger.info(f"   {winner_name}收取{collected_score}分平洞分数")

            return {
                "final_score_p1": final_p1,
                "final_score_p2": final_p2,
                "tie_info": {
                    "is_tie": False,
                    "collected_score": collected_score,
                    "remaining_ties": state["tie_count"],
                    "remaining_score": state["tie_accumulated_score"],
                },
            }

    def _get_tie_score(self, tie_rule: str) -> int:
        """获取平洞累积分数"""
        if tie_rule == "平洞算1点":
            return 1
        elif tie_rule == "平洞翻倍(不算鸟鹰奖)":
            return 0  # 翻倍模式不累积固定分数，而是累积次数
        return 0

    def _collect_tie_score(self, base_p1: float, base_p2: float, p1_raw: int, p2_raw: int, par: int,
                           state: Dict[str, Any], config: GameConfig, p1_name: str, p2_name: str) -> float:
        """收取平洞分数（固定规则：par收1/鸟收2/鹰收5）"""
        winner_raw = p1_raw if base_p1 > base_p2 else p2_raw
        winner_name = p1_name if base_p1 > base_p2 else p2_name
        score_to_par = winner_raw - par

        # 根据获胜者成绩确定收取次数
        collect_count = 1  # par收1
        achievement = "标准杆"
        if score_to_par == -1:  # 小鸟收2
            collect_count = 2
            achievement = "小鸟"
        elif score_to_par <= -2:  # 老鹰及以上收5
            collect_count = 5
            achievement = self._get_achievement_name(score_to_par, winner_raw, par)
        elif score_to_par == 0:
            achievement = "标准杆"
        else:
            achievement = f"+{score_to_par}杆"

        # 实际收取次数不能超过累积次数
        actual_collect = min(collect_count, state["tie_count"])

        self.logger.info(f"   平洞收取计算:")
        self.logger.info(f"     获胜者{winner_name}打出{achievement}，期望收取{collect_count}次平洞")
        self.logger.info(f"     实际可收取: {actual_collect}次 (当前累积{state['tie_count']}次)")

        if config.tie_scoring_rule == "平洞翻倍(不算鸟鹰奖)":
            # 翻倍模式：收取分数 = 本洞胜负差 × 收取次数
            base_score = abs(base_p1 - base_p2)
            collected = base_score * actual_collect
            self.logger.info(f"     翻倍模式: 本洞胜负差{base_score} × {actual_collect}次 = {collected}分")
        else:
            # 固定分数模式：收取对应数量的累积分数
            collected = min(actual_collect, state["tie_accumulated_score"])
            self.logger.info(f"     固定分数模式: 收取{collected}分 (每次1分)")
            state["tie_accumulated_score"] -= collected

        state["tie_count"] -= actual_collect

        self.logger.info(f"     收取后剩余: {state['tie_count']}次/{state.get('tie_accumulated_score', 0)}分")

        return collected

    def _calculate_special_achievements(self, hole: Hole, player_scores: Dict[str, int]) -> List[Dict[str, Any]]:
        """计算特殊成就"""
        achievements = []
        for pid, raw in player_scores.items():
            score_to_par = raw - hole.par
            achievement_name = self._get_achievement_name(score_to_par, raw, hole.par)
            if score_to_par <= 0:  # 只记录Par及以上成绩
                achievements.append({
                    "player_id": pid,
                    "achievement": achievement_name,
                    "score_to_par": score_to_par,
                    "raw_score": raw
                })
        return achievements

    def _get_achievement_name(self, score_to_par: int, raw: Optional[int] = None, par: Optional[int] = None) -> str:
        """获取成就名称"""
        if par is not None and raw is not None and par == 3 and raw == 1:
            return "一杆进洞"

        mapping = {
            -3: "信天翁", -2: "老鹰", -1: "小鸟", 0: "标准杆",
            1: "柏忌", 2: "双柏忌", 3: "三柏忌"
        }
        return mapping.get(score_to_par, f"+{score_to_par}杆")

    def _update_score_state(self, score_state: Dict[str, Any], hole_result: Dict[str, Any]):
        """更新分数状态"""
        for pr in hole_result["pairing_results"]:
            key = pr["pairing"]
            state = score_state["pairing_states"][key]
            state["player1_total"] += pr["final_scores"]["player1"]
            state["player2_total"] += pr["final_scores"]["player2"]

    def _calculate_final_summary(self, game_data: StrokeMatchGameData, score_state: Dict[str, Any],
                                 hole_details: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算最终汇总"""
        self.logger.info(f"\n{'=' * 80}")
        self.logger.info(f"最终总分汇总")
        self.logger.info(f"{'=' * 80}")

        player_totals: Dict[str, float] = {p.id: 0 for p in game_data.players}
        for key, st in score_state["pairing_states"].items():
            p1, p2 = key.split("_vs_")
            player_totals[p1] += st["player1_total"]
            player_totals[p2] += st["player2_total"]

        final_scores = [
            {"player_id": p.id, "player_name": p.name, "total_score": player_totals[p.id]}
            for p in game_data.players
        ]

        # 打印每个配对的分数详情
        self.logger.info(f"\n配对分数详情:")
        for pairing in game_data.game_config.pairings:
            key = f"{pairing.player1_id}_vs_{pairing.player2_id}"
            st = score_state["pairing_states"][key]

            p1_name = next(p.name for p in game_data.players if p.id == pairing.player1_id)
            p2_name = next(p.name for p in game_data.players if p.id == pairing.player2_id)

            self.logger.info(f"  {p1_name} vs {p2_name}:")
            self.logger.info(f"    {p1_name}: {st['player1_total']:+.1f}分")
            self.logger.info(f"    {p2_name}: {st['player2_total']:+.1f}分")
            if st['tie_count'] > 0 or st.get('tie_accumulated_score', 0) > 0:
                self.logger.info(f"    剩余平洞: {st['tie_count']}次/{st.get('tie_accumulated_score', 0)}分")

        # 打印最终总分排名
        self.logger.info(f"\n最终总分排名:")
        sorted_scores = sorted(final_scores, key=lambda x: x["total_score"], reverse=True)
        for i, score in enumerate(sorted_scores, 1):
            self.logger.info(f"  {i}. {score['player_name']}: {score['total_score']:+.1f}分")

        # 验证零和游戏
        total_sum = sum(s["total_score"] for s in final_scores)
        if abs(total_sum) < 0.001:  # 考虑浮点误差
            self.logger.info(f"\n✓ 零和游戏验证通过 (总和: {total_sum})")
        else:
            self.logger.warning(f"\n⚠ 零和游戏验证失败 (总和: {total_sum})")

        pairing_summaries = []
        for pairing in game_data.game_config.pairings:
            key = f"{pairing.player1_id}_vs_{pairing.player2_id}"
            st = score_state["pairing_states"][key]
            pairing_summaries.append({
                "pairing": key,
                "player1_id": pairing.player1_id,
                "player2_id": pairing.player2_id,
                "player1_total": st["player1_total"],
                "player2_total": st["player2_total"],
                "remaining_ties": st["tie_count"],
                "remaining_tie_score": st.get("tie_accumulated_score", 0),
            })

        return {
            "total_holes": len(game_data.holes),
            "game_mode": game_data.game_config.mode,
            "tie_scoring_rule": game_data.game_config.tie_scoring_rule,
            # 修复：使用正确的字段名
            "bird_eagle_reward": game_data.game_config.bird_eagle_reward,
            "players": [{"id": p.id, "name": p.name} for p in game_data.players],
            "pairings": pairing_summaries,
            "final_scores": final_scores,
        }

    def _error_response(self, message: str) -> Dict[str, Any]:
        return {
            "success": False,
            "error_code": "CALCULATION_ERROR",
            "message": message,
            "data": None
        }


# 对外函数
def calculate_stroke_match_score(game_data: StrokeMatchGameData) -> Dict[str, Any]:
    """更新后的挂杆挂洞计分接口"""
    try:
        api = UpdatedStrokeMatchAPI()
        result = api.calculate_stroke_match_score(game_data)
        if result.get("success"):
            pass
        else:
            logger.error(f"挂杆挂洞计分失败: {result.get('message')}")
        return result
    except Exception as e:
        logger.error(f"挂杆挂洞计分接口异常: {str(e)}")
        return {
            "success": False,
            "error_code": "INTERFACE_ERROR",
            "message": f"接口调用异常: {str(e)}",
            "data": None
        }
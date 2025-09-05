import json
import copy
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from utils.nlp_logging import CustomLogger

"""
拉丝三点高尔夫赌球游戏计分接口
实现完整的拉丝三点游戏计分逻辑，包括：
- 固拉/乱拉两种模式
- 高手不见面配置（新增）
- 组合PK相加/相乘模式（新增）
- 三项比较计分（最好/最差/总分）
- 鸟鹰基础分数和额外奖励系统
- 双杀奖励机制
- 平洞累积和收取规则
- 包赔规则判定和分数重分配
- 复杂让杆规则和条件限制
- 捐锅和让分最终结算

"""

logger = CustomLogger(name="DaoTest 3-Point Las Vegas api", write_to_file=True)

# 鸟鹰基础分数系统（基于原始杆数判定）
BIRD_EAGLE_SCORES = {
    -3: 32,  # 信天翁
    -2: 16,  # 老鹰
    -1: 8,  # 小鸟
    0: 4,  # 标准杆
    1: 2,  # 柏忌
    2: 1,  # 双柏忌
    3: 0,  # 三柏忌
    4: -1  # 四柏忌及以上
}


# 数据模型定义
class DialogueMessage(BaseModel):
    role: str
    content: str


class GameConfig(BaseModel):
    mode: str = Field(..., description="游戏模式: 固拉 | 乱拉")
    base_score: int = Field(default=1, description="基础分")

    # 新增：组合PK配置
    combination_pk_config: Dict[str, str] = Field(default_factory=lambda: {
        "mode": "双方总杆相加PK"  # "双方总杆相加PK" | "双方总杆相乘PK"
    })

    # 新增：高手不见面配置（仅乱拉模式）
    expert_separation_config: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": False,  # 是否启用高手不见面
        "expert_players": []  # 高手玩家ID列表，必须2人
    })

    double_kill_config: Dict[str, Any] = Field(default_factory=lambda: {"type": "不奖励"})

    bird_eagle_config: Dict[str, Any] = Field(default_factory=lambda: {
        "condition": "合并pk赢了才奖励",
        "extra_reward": "不奖励"
    })

    tie_config: Dict[str, Any] = Field(default_factory=lambda: {
        "definition": "得分差为0",
        "scoring": "平洞跳过(无肉)",
        "collect_rule": "赢了全收掉"
    })

    compensation_config: Dict[str, Any] = Field(default_factory=lambda: {
        "scope": "不包赔",
        "conditions": {
            "double_par": False,
            "plus_three": False,
            "diff_three": False
        }
    })

    donation_config: Dict[str, str] = Field(default_factory=lambda: {"type": "不捐"})

    handicap_config: Dict[str, Any] = Field(default_factory=lambda: {
        "restrictions": {
            "only_total_pk": False,
            "no_leader": False,
            "no_par_bird_eagle": False
        }
    })

    score_adjustment_config: Dict[str, Any] = Field(default_factory=lambda: {
        "mode": "单让",
        "adjustment_type": "实让",
        "points": 0
    })

    # 移入GameConfig：让杆设置
    handicap_settings: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="让杆设置")


class Player(BaseModel):
    id: str
    name: str


class FixedTeam(BaseModel):
    team_id: int
    players: List[str]


class Score(BaseModel):
    player_id: str
    raw_strokes: int


class Hole(BaseModel):
    hole_number: int
    par: int
    hole_type: str = Field(..., description="三杆洞 | 四杆洞 | 五杆洞")
    tee_order: List[str]
    scores: List[Score]


class LaSiGameData(BaseModel):
    game_config: GameConfig
    players: List[Player]
    fixed_teams: Optional[List[FixedTeam]] = None
    holes: List[Hole]


class LaSiThreePointAPI:
    def __init__(self):
        self.logger = logger

    def calculate_lasi_score(self, game_data: LaSiGameData) -> Dict[str, Any]:
        """拉丝三点计分主函数"""
        try:
            self.logger.info("=== 开始拉丝三点计分 ===")

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

            self.logger.info("=== 拉丝三点计分完成 ===")
            return result

        except Exception as e:
            self.logger.error(f"计分过程发生错误: {str(e)}")
            return self._error_response(f"计分失败: {str(e)}")

    def _validate_game_data(self, game_data: LaSiGameData) -> bool:
        """验证游戏数据"""
        try:
            # 验证玩家数量
            if len(game_data.players) != 4:
                self.logger.error(f"玩家人数错误: {len(game_data.players)}")
                return False

            # 验证固拉模式的队伍配置
            if game_data.game_config.mode == "固拉":
                if not game_data.fixed_teams or len(game_data.fixed_teams) != 2:
                    self.logger.error("固拉模式缺少队伍配置")
                    return False

            # 验证乱拉模式的高手不见面配置
            if game_data.game_config.mode == "乱拉":
                expert_config = game_data.game_config.expert_separation_config
                if expert_config.get("enabled", False):
                    expert_players = expert_config.get("expert_players", [])
                    if len(expert_players) != 2:
                        self.logger.error("高手不见面模式必须选择2名高手")
                        return False

                    # 验证高手玩家ID是否存在
                    player_ids = [p.id for p in game_data.players]
                    for expert_id in expert_players:
                        if expert_id not in player_ids:
                            self.logger.error(f"高手玩家ID不存在: {expert_id}")
                            return False

            # 验证洞次数据
            if not game_data.holes:
                self.logger.error("缺少洞次数据")
                return False

            for hole in game_data.holes:
                if len(hole.scores) != 4:
                    self.logger.error(f"第{hole.hole_number}洞分数数据不完整")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"数据验证异常: {str(e)}")
            return False

    def _initialize_score_state(self, game_data: LaSiGameData) -> Dict[str, Any]:
        """初始化计分状态"""
        player_ids = [p.id for p in game_data.players]

        return {
            "player_total_scores": {pid: 0 for pid in player_ids},
            "tie_accumulated_score": 0,
            "tie_count": 0,
            "current_tee_order": []
        }

    def _process_hole(self, hole: Hole, game_data: LaSiGameData, score_state: Dict[str, Any]) -> Dict[str, Any]:
        """处理单洞计分"""
        self.logger.info(f"开始处理第{hole.hole_number}洞")

        try:
            # 1. 确定队伍配对
            teams = self._determine_teams(hole, game_data)

            # 2. 计算净杆数（应用让杆）
            net_scores = self._calculate_net_scores(hole, game_data, teams)

            # 3. 执行三项比较PK
            pk_results = self._execute_pk_comparisons(teams, net_scores, game_data.game_config)

            # 4. 判定特殊奖励
            special_rewards = self._calculate_special_rewards(hole, teams, net_scores, pk_results,
                                                              game_data.game_config)

            # 5. 处理包赔逻辑
            compensation = self._handle_compensation(hole, teams, net_scores, game_data.game_config)

            # 6. 处理平洞逻辑
            tie_status = self._handle_tie_logic(pk_results, score_state, game_data.game_config)

            # 7. 计算最终洞次得分
            hole_scores = self._calculate_hole_scores(teams, pk_results, special_rewards, compensation, tie_status,
                                                      game_data.game_config)

            # 8. 确定下洞开球顺序
            next_tee_order = self._determine_next_tee_order(hole, teams, net_scores)
            score_state["current_tee_order"] = next_tee_order

            return {
                "hole_number": hole.hole_number,
                "par": hole.par,
                "hole_type": hole.hole_type,
                "teams": teams,
                "pk_results": pk_results,
                "special_rewards": special_rewards,
                "compensation": compensation,
                "tie_status": tie_status,
                "hole_scores": hole_scores,
                "next_tee_order": next_tee_order
            }

        except Exception as e:
            self.logger.error(f"处理第{hole.hole_number}洞时出错: {str(e)}")
            raise

    def _determine_teams(self, hole: Hole, game_data: LaSiGameData) -> List[Dict[str, Any]]:
        """确定队伍配对"""
        if game_data.game_config.mode == "固拉":
            # 固拉模式：使用固定队伍
            teams = []
            for fixed_team in game_data.fixed_teams:
                team_data = {
                    "team_id": fixed_team.team_id,
                    "team_players": fixed_team.players,
                    "raw_scores": [],
                    "handicaps": [],
                    "net_scores": []
                }
                teams.append(team_data)
        else:
            # 乱拉模式：根据开球顺序和高手不见面规则配对
            teams = self._determine_luanla_teams(hole, game_data)

        # 填充杆数数据
        score_map = {score.player_id: score.raw_strokes for score in hole.scores}

        for team in teams:
            for player_id in team["team_players"]:
                raw_score = score_map.get(player_id, 0)
                team["raw_scores"].append(raw_score)

        return teams

    def _determine_luanla_teams(self, hole: Hole, game_data: LaSiGameData) -> List[Dict[str, Any]]:
        """确定乱拉模式的队伍配对"""
        tee_order = hole.tee_order
        expert_config = game_data.game_config.expert_separation_config

        if expert_config.get("enabled", False):
            # 高手不见面模式
            expert_players = expert_config.get("expert_players", [])

            # 验证高手是否都在开球顺序中
            if not all(expert in tee_order for expert in expert_players):
                self.logger.warning("高手玩家不在开球顺序中，使用默认配对")
                return self._default_luanla_pairing(tee_order)

            # 确保高手分在不同队伍
            expert1, expert2 = expert_players[0], expert_players[1]
            other_players = [p for p in tee_order if p not in expert_players]

            # 将高手分别放在队伍1和队伍2
            team1_players = [expert1, other_players[0]]
            team2_players = [expert2, other_players[1]]

            teams = [
                {
                    "team_id": 1,
                    "team_players": team1_players,
                    "raw_scores": [],
                    "handicaps": [],
                    "net_scores": []
                },
                {
                    "team_id": 2,
                    "team_players": team2_players,
                    "raw_scores": [],
                    "handicaps": [],
                    "net_scores": []
                }
            ]

            self.logger.info(f"高手不见面配对: 队伍1 {team1_players}, 队伍2 {team2_players}")

        else:
            # 默认乱拉配对：第1&第4 vs 第2&第3
            teams = self._default_luanla_pairing(tee_order)

        return teams

    def _default_luanla_pairing(self, tee_order: List[str]) -> List[Dict[str, Any]]:
        """默认乱拉配对模式"""
        return [
            {
                "team_id": 1,
                "team_players": [tee_order[0], tee_order[3]],  # 第1和第4
                "raw_scores": [],
                "handicaps": [],
                "net_scores": []
            },
            {
                "team_id": 2,
                "team_players": [tee_order[1], tee_order[2]],  # 第2和第3
                "raw_scores": [],
                "handicaps": [],
                "net_scores": []
            }
        ]

    def _calculate_net_scores(self, hole: Hole, game_data: LaSiGameData, teams: List[Dict[str, Any]]) -> Dict[
        str, float]:
        """计算净杆数（应用让杆）"""
        net_scores = {}

        for team in teams:
            for i, player_id in enumerate(team["team_players"]):
                raw_score = team["raw_scores"][i]

                # 获取让杆数 - 修改为使用game_config.handicap_settings
                handicap = self._get_player_handicap(player_id, hole.hole_type, game_data.game_config.handicap_settings)

                # 检查让杆限制条件
                effective_handicap = self._apply_handicap_restrictions(
                    player_id, handicap, raw_score, hole.par, teams, game_data.game_config
                )

                net_score = raw_score - effective_handicap

                team["handicaps"].append(effective_handicap)
                team["net_scores"].append(net_score)
                net_scores[player_id] = net_score

        # 计算队伍统计（根据组合PK模式）
        for team in teams:
            team["best_score"] = min(team["net_scores"])
            team["worst_score"] = max(team["net_scores"])

            # 根据组合PK配置计算总分
            combination_mode = game_data.game_config.combination_pk_config.get("mode", "双方总杆相加PK")
            if combination_mode == "双方总杆相乘PK":
                team["total_score"] = team["net_scores"][0] * team["net_scores"][1]
            else:  # 默认相加
                team["total_score"] = sum(team["net_scores"])

        return net_scores

    def _get_player_handicap(self, player_id: str, hole_type: str,
                             handicap_settings: Dict[str, Dict[str, float]]) -> float:
        """获取选手让杆数"""
        if player_id in handicap_settings:
            return handicap_settings[player_id].get(hole_type, 0.0)
        return 0.0

    def _apply_handicap_restrictions(self, player_id: str, handicap: float, raw_score: int, par: int,
                                     teams: List[Dict[str, Any]], game_config: GameConfig) -> float:
        """应用让杆限制条件"""
        restrictions = game_config.handicap_config.get("restrictions", {})

        # 检查是否为该洞成绩最好的选手（打头不让）
        if restrictions.get("no_leader", False):
            all_raw_scores = []
            for team in teams:
                all_raw_scores.extend(team["raw_scores"])

            if raw_score == min(all_raw_scores):
                self.logger.info(f"选手{player_id}打头不让杆")
                return 0.0

        # 检查是否打出Par/鸟/鹰（不让杆）
        if restrictions.get("no_par_bird_eagle", False):
            score_to_par = raw_score - par
            if score_to_par <= 0:  # Par或更好
                self.logger.info(f"选手{player_id}打出{score_to_par}杆，不让杆")
                return 0.0

        return handicap

    def _execute_pk_comparisons(self, teams: List[Dict[str, Any]], net_scores: Dict[str, float],
                                game_config: GameConfig) -> Dict[str, Any]:
        """执行三项比较PK"""
        team1, team2 = teams[0], teams[1]
        base_score = game_config.base_score

        # 最好成绩PK
        best_pk = self._compare_scores(team1["best_score"], team2["best_score"], base_score)

        # 最差成绩PK
        worst_pk = self._compare_scores(team1["worst_score"], team2["worst_score"], base_score)

        # 总分PK (现在支持相加和相乘两种模式)
        total_pk = self._compare_scores(team1["total_score"], team2["total_score"], base_score)

        # 添加组合PK模式信息
        combination_mode = game_config.combination_pk_config.get("mode", "双方总杆相加PK")
        total_pk["combination_mode"] = combination_mode

        return {
            "best_pk": best_pk,
            "worst_pk": worst_pk,
            "total_pk": total_pk
        }

    def _compare_scores(self, team1_score: float, team2_score: float, base_score: int) -> Dict[str, Any]:
        """比较两队得分"""
        if team1_score < team2_score:
            return {
                "winner_team": 1,
                "score_diff": base_score,
                "team1_score": team1_score,
                "team2_score": team2_score
            }
        elif team1_score > team2_score:
            return {
                "winner_team": 2,
                "score_diff": base_score,
                "team1_score": team1_score,
                "team2_score": team2_score
            }
        else:
            return {
                "winner_team": 0,  # 平局
                "score_diff": 0,
                "team1_score": team1_score,
                "team2_score": team2_score
            }

    def _calculate_special_rewards(self, hole: Hole, teams: List[Dict[str, Any]], net_scores: Dict[str, float],
                                   pk_results: Dict[str, Any], game_config: GameConfig) -> Dict[str, Any]:
        """计算特殊奖励"""
        # 双杀奖励
        double_kill = self._check_double_kill(teams, game_config)

        # 鸟鹰奖励
        bird_eagle = self._calculate_bird_eagle_rewards(hole, teams, pk_results, game_config)

        return {
            "double_kill": double_kill,
            "bird_eagle": bird_eagle
        }

    def _check_double_kill(self, teams: List[Dict[str, Any]], game_config: GameConfig) -> Dict[str, Any]:
        """检查双杀奖励"""
        team1, team2 = teams[0], teams[1]

        # 双杀条件：一队最差成绩比对方最好成绩还要好（基于净杆数）
        if team1["worst_score"] < team2["best_score"]:
            winner_team = 1
        elif team2["worst_score"] < team1["best_score"]:
            winner_team = 2
        else:
            return {"triggered": False}

        # 获取双杀奖励分数
        double_kill_config = game_config.double_kill_config.get("type", "不奖励")
        reward_score = self._parse_double_kill_reward(double_kill_config)

        return {
            "triggered": True,
            "winner_team": winner_team,
            "reward_score": reward_score
        }

    def _parse_double_kill_reward(self, config: str) -> int:
        """解析双杀奖励分数"""
        if config == "不奖励":
            return 0
        elif config == "奖励1分":
            return 1
        elif config == "奖励2分":
            return 2
        elif config == "奖励3分":
            return 3
        elif config == "翻倍奖励":
            return 2  # 翻倍的具体实现需要根据上下文
        else:
            return 0

    def _calculate_bird_eagle_rewards(self, hole: Hole, teams: List[Dict[str, Any]], pk_results: Dict[str, Any],
                                      game_config: GameConfig) -> List[Dict[str, Any]]:
        """计算鸟鹰奖励"""
        rewards = []

        # 检查鸟鹰奖励条件
        condition = game_config.bird_eagle_config.get("condition", "合并pk赢了才奖励")

        for team in teams:
            for i, player_id in enumerate(team["team_players"]):
                raw_score = team["raw_scores"][i]
                score_to_par = raw_score - hole.par

                # 获取基础分数（基于原始杆数）
                base_score = BIRD_EAGLE_SCORES.get(score_to_par, 0)
                if score_to_par > 4:
                    base_score = BIRD_EAGLE_SCORES[4]  # 四柏忌及以上

                # 获取额外奖励
                extra_reward = self._get_extra_bird_eagle_reward(score_to_par, game_config)

                # 检查是否符合奖励条件
                eligible = self._check_bird_eagle_eligibility(team["team_id"], pk_results, condition)

                achievement_name = self._get_achievement_name(score_to_par)

                rewards.append({
                    "player_id": player_id,
                    "achievement": achievement_name,
                    "base_score": base_score,
                    "extra_reward": extra_reward,
                    "eligible": eligible,
                    "applied": eligible
                })

        return rewards

    def _get_extra_bird_eagle_reward(self, score_to_par: int, game_config: GameConfig) -> int:
        """获取鸟鹰额外奖励"""
        extra_reward_config = game_config.bird_eagle_config.get("extra_reward", "不奖励")

        if extra_reward_config == "不奖励":
            return 0

        # 解析额外奖励配置
        reward_map = {
            "鸟+1/鹰+4/HIO+9": {-1: 1, -2: 4, -3: 9},
            "鸟+1/鹰+4/HIO+8": {-1: 1, -2: 4, -3: 8},
            "鸟+1/鹰+5/HIO+10": {-1: 1, -2: 5, -3: 10},
            "鸟+1/鹰+10/HIO+20": {-1: 1, -2: 10, -3: 20},
            "鸟+2/鹰+4/HIO+8": {-1: 2, -2: 4, -3: 8},
            "鸟+2/鹰+5/HIO+10": {-1: 2, -2: 5, -3: 10}
        }

        if extra_reward_config in reward_map:
            return reward_map[extra_reward_config].get(score_to_par, 0)

        return 0

    def _check_bird_eagle_eligibility(self, team_id: int, pk_results: Dict[str, Any], condition: str) -> bool:
        """检查鸟鹰奖励资格"""
        if condition == "合并pk赢了才奖励":
            # 计算该队伍的总得分
            team_total = 0
            for pk_name, pk_result in pk_results.items():
                if pk_result["winner_team"] == team_id:
                    team_total += pk_result["score_diff"]
                elif pk_result["winner_team"] != 0:  # 不是平局
                    team_total -= pk_result["score_diff"]

            return team_total > 0

        return True  # 其他条件默认符合

    def _get_achievement_name(self, score_to_par: int) -> str:
        """获取成就名称"""
        names = {
            -3: "信天翁",
            -2: "老鹰",
            -1: "小鸟",
            0: "标准杆",
            1: "柏忌",
            2: "双柏忌",
            3: "三柏忌"
        }

        if score_to_par >= 4:
            return "四柏忌及以上"

        return names.get(score_to_par, "未知")

    def _handle_compensation(self, hole: Hole, teams: List[Dict[str, Any]], net_scores: Dict[str, float],
                             game_config: GameConfig) -> Dict[str, Any]:
        """处理包赔逻辑"""
        compensation_config = game_config.compensation_config

        if compensation_config.get("scope") == "不包赔":
            return {"triggered": False, "details": []}

        compensation_details = []
        conditions = compensation_config.get("conditions", {})

        for team in teams:
            for i, player_id in enumerate(team["team_players"]):
                net_score = team["net_scores"][i]
                teammate_net_score = team["net_scores"][1 - i]  # 队友净杆数

                # 检查包赔条件
                should_compensate = False
                reason = ""

                if conditions.get("double_par", False):
                    if net_score >= hole.par * 2:
                        should_compensate = True
                        reason = "双par及以上"

                if conditions.get("plus_three", False):
                    if net_score >= hole.par + 3:
                        should_compensate = True
                        reason = "+3及以上"

                if conditions.get("diff_three", False):
                    if abs(net_score - teammate_net_score) >= 3:
                        should_compensate = True
                        reason = "与队友相差3杆及以上"

                if should_compensate:
                    compensation_details.append({
                        "player_id": player_id,
                        "reason": reason,
                        "net_score": net_score,
                        "teammate_score": teammate_net_score
                    })

        return {
            "triggered": len(compensation_details) > 0,
            "details": compensation_details
        }

    def _handle_tie_logic(self, pk_results: Dict[str, Any], score_state: Dict[str, Any], game_config: GameConfig) -> \
    Dict[str, Any]:
        """处理平洞逻辑"""
        tie_config = game_config.tie_config

        # 检查是否为平洞
        is_tie = self._check_tie_condition(pk_results, tie_config)

        if is_tie:
            # 获取平洞分数
            tie_score = self._get_tie_score(tie_config)
            score_state["tie_accumulated_score"] += tie_score
            score_state["tie_count"] += 1

            return {
                "is_tie": True,
                "accumulated_score": score_state["tie_accumulated_score"],
                "tie_count": score_state["tie_count"]
            }
        else:
            # 不是平洞，检查是否需要收取累积分数
            collected_score = 0
            if score_state["tie_accumulated_score"] > 0:
                collected_score = self._collect_tie_score(pk_results, score_state, tie_config)

            return {
                "is_tie": False,
                "accumulated_score": score_state["tie_accumulated_score"],
                "collected_score": collected_score
            }

    def _check_tie_condition(self, pk_results: Dict[str, Any], tie_config: Dict[str, Any]) -> bool:
        """检查平洞条件"""
        definition = tie_config.get("definition", "得分差为0")

        if definition == "得分差为0":
            # 三项比较结果完全相同
            total_diff = 0
            for pk_result in pk_results.values():
                if pk_result["winner_team"] == 1:
                    total_diff += pk_result["score_diff"]
                elif pk_result["winner_team"] == 2:
                    total_diff -= pk_result["score_diff"]

            return total_diff == 0

        # 其他平洞定义的实现...
        return False

    def _get_tie_score(self, tie_config: Dict[str, Any]) -> int:
        """获取平洞分数"""
        scoring = tie_config.get("scoring", "平洞跳过(无肉)")

        if scoring == "平洞跳过(无肉)":
            return 0
        elif scoring == "平洞算1点":
            return 1
        elif scoring == "平洞算2点":
            return 2
        elif scoring == "平洞算3点":
            return 3
        elif scoring == "平洞算4点":
            return 4

        return 0

    def _collect_tie_score(self, pk_results: Dict[str, Any], score_state: Dict[str, Any],
                           tie_config: Dict[str, Any]) -> int:
        """收取平洞分数"""
        collect_rule = tie_config.get("collect_rule", "赢了全收掉")

        if collect_rule == "赢了全收掉":
            collected = score_state["tie_accumulated_score"]
            score_state["tie_accumulated_score"] = 0
            score_state["tie_count"] = 0
            return collected

        # 其他收取规则的实现...
        return 0

    def _calculate_hole_scores(self, teams: List[Dict[str, Any]], pk_results: Dict[str, Any],
                               special_rewards: Dict[str, Any], compensation: Dict[str, Any],
                               tie_status: Dict[str, Any], game_config: GameConfig) -> List[Dict[str, Any]]:
        """计算最终洞次得分"""
        hole_scores = []

        # 初始化每个选手的得分
        for team in teams:
            for player_id in team["team_players"]:
                hole_scores.append({
                    "player_id": player_id,
                    "base_score": 0,
                    "bird_eagle_score": 0,
                    "extra_reward": 0,
                    "double_kill_score": 0,
                    "tie_score": 0,
                    "final_score": 0
                })

        # 计算基础PK得分
        self._apply_base_pk_scores(hole_scores, teams, pk_results)

        # 应用鸟鹰分数和奖励
        self._apply_bird_eagle_scores(hole_scores, special_rewards["bird_eagle"])

        # 应用双杀奖励
        self._apply_double_kill_reward(hole_scores, teams, special_rewards["double_kill"])

        # 处理包赔调整
        self._apply_compensation_adjustment(hole_scores, teams, compensation)

        # 应用平洞分数
        self._apply_tie_scores(hole_scores, teams, tie_status, pk_results)

        # 计算最终分数
        for score in hole_scores:
            score["final_score"] = (
                    score["base_score"] +
                    score["bird_eagle_score"] +
                    score["extra_reward"] +
                    score["double_kill_score"] +
                    score["tie_score"]
            )

        return hole_scores

    def _apply_base_pk_scores(self, hole_scores: List[Dict[str, Any]], teams: List[Dict[str, Any]],
                              pk_results: Dict[str, Any]):
        """应用基础PK得分"""
        # 计算每队在三项比较中的总得分
        team_base_scores = {1: 0, 2: 0}

        for pk_result in pk_results.values():
            winner_team = pk_result["winner_team"]
            if winner_team != 0:  # 不是平局
                team_base_scores[winner_team] += pk_result["score_diff"]
                other_team = 2 if winner_team == 1 else 1
                team_base_scores[other_team] -= pk_result["score_diff"]

        # 分配给每个选手
        for team in teams:
            team_score = team_base_scores[team["team_id"]]
            for player_id in team["team_players"]:
                for score in hole_scores:
                    if score["player_id"] == player_id:
                        score["base_score"] = team_score
                        break

    def _apply_bird_eagle_scores(self, hole_scores: List[Dict[str, Any]], bird_eagle_rewards: List[Dict[str, Any]]):
        """应用鸟鹰分数和奖励"""
        for reward in bird_eagle_rewards:
            player_id = reward["player_id"]
            for score in hole_scores:
                if score["player_id"] == player_id:
                    if reward["applied"]:
                        score["bird_eagle_score"] = reward["base_score"]
                        score["extra_reward"] = reward["extra_reward"]
                    break

    def _apply_double_kill_reward(self, hole_scores: List[Dict[str, Any]], teams: List[Dict[str, Any]],
                                  double_kill: Dict[str, Any]):
        """应用双杀奖励"""
        if not double_kill.get("triggered", False):
            return

        winner_team_id = double_kill["winner_team"]
        reward_score = double_kill["reward_score"]

        # 找到获胜队伍，给每个成员加分
        for team in teams:
            if team["team_id"] == winner_team_id:
                for player_id in team["team_players"]:
                    for score in hole_scores:
                        if score["player_id"] == player_id:
                            score["double_kill_score"] = reward_score
                            break
            else:
                # 失败队伍扣分
                for player_id in team["team_players"]:
                    for score in hole_scores:
                        if score["player_id"] == player_id:
                            score["double_kill_score"] = -reward_score
                            break

    def _apply_compensation_adjustment(self, hole_scores: List[Dict[str, Any]], teams: List[Dict[str, Any]],
                                       compensation: Dict[str, Any]):
        """应用包赔调整"""
        if not compensation.get("triggered", False):
            return

        for detail in compensation["details"]:
            compensating_player = detail["player_id"]

            # 找到包赔选手的队友
            teammate_id = None
            for team in teams:
                if compensating_player in team["team_players"]:
                    for player_id in team["team_players"]:
                        if player_id != compensating_player:
                            teammate_id = player_id
                            break
                    break

            if teammate_id:
                # 找到队友的负分，由包赔选手承担
                teammate_score = None
                compensating_score = None

                for score in hole_scores:
                    if score["player_id"] == teammate_id:
                        teammate_score = score
                    elif score["player_id"] == compensating_player:
                        compensating_score = score

                if teammate_score and compensating_score:
                    # 计算队友的负分总和
                    teammate_negative = min(0, teammate_score["base_score"] +
                                            teammate_score["bird_eagle_score"] +
                                            teammate_score["extra_reward"] +
                                            teammate_score["double_kill_score"])

                    if teammate_negative < 0:
                        # 包赔选手承担队友的负分
                        compensating_score["base_score"] += teammate_negative
                        teammate_score["base_score"] -= teammate_negative

                        self.logger.info(f"选手{compensating_player}包赔队友{teammate_id}的{-teammate_negative}分")

    def _apply_tie_scores(self, hole_scores: List[Dict[str, Any]], teams: List[Dict[str, Any]],
                          tie_status: Dict[str, Any], pk_results: Dict[str, Any]):
        """应用平洞分数"""
        if tie_status.get("is_tie", False):
            # 本洞平洞，不分配分数
            return

        collected_score = tie_status.get("collected_score", 0)
        if collected_score > 0:
            # 分配收取的平洞分数给获胜方
            total_winner_score = 0
            for pk_result in pk_results.values():
                if pk_result["winner_team"] == 1:
                    total_winner_score += pk_result["score_diff"]
                elif pk_result["winner_team"] == 2:
                    total_winner_score -= pk_result["score_diff"]

            if total_winner_score > 0:
                winner_team_id = 1
            elif total_winner_score < 0:
                winner_team_id = 2
            else:
                return  # 平局不分配

            # 给获胜队伍每人加平洞分
            for team in teams:
                multiplier = collected_score if team["team_id"] == winner_team_id else -collected_score
                for player_id in team["team_players"]:
                    for score in hole_scores:
                        if score["player_id"] == player_id:
                            score["tie_score"] = multiplier
                            break

    def _determine_next_tee_order(self, hole: Hole, teams: List[Dict[str, Any]], net_scores: Dict[str, float]) -> List[
        str]:
        """确定下洞开球顺序"""
        # 按净杆数排序，最好成绩先开球
        player_scores = []
        for team in teams:
            for i, player_id in enumerate(team["team_players"]):
                player_scores.append((player_id, team["net_scores"][i]))

        # 按净杆数升序排序
        player_scores.sort(key=lambda x: x[1])

        return [player_id for player_id, _ in player_scores]

    def _update_score_state(self, score_state: Dict[str, Any], hole_result: Dict[str, Any]):
        """更新累积计分状态"""
        # 更新选手总分
        for score in hole_result["hole_scores"]:
            player_id = score["player_id"]
            score_state["player_total_scores"][player_id] += score["final_score"]

    def _calculate_final_summary(self, game_data: LaSiGameData, score_state: Dict[str, Any],
                                 hole_details: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算最终总结"""
        # 基础总分
        final_scores = []
        for player in game_data.players:
            total_score = score_state["player_total_scores"][player.id]
            final_scores.append({
                "player_id": player.id,
                "player_name": player.name,
                "total_score": total_score,
                "after_donation": total_score,  # 稍后计算捐锅
                "after_adjustment": total_score  # 稍后计算让分
            })

        # 计算捐锅
        donation_summary = self._calculate_donation(final_scores, game_data.game_config)

        # 应用捐锅
        for score in final_scores:
            donated = 0
            for detail in donation_summary["donation_details"]:
                if detail["player_id"] == score["player_id"]:
                    donated = detail["donated"]
                    break
            score["after_donation"] = score["total_score"] - donated
            score["after_adjustment"] = score["after_donation"]  # 暂时相等

        # 计算让分调整
        adjustment_summary = self._calculate_score_adjustment(final_scores, game_data.game_config)

        return {
            "total_holes": len(game_data.holes),
            "game_mode": game_data.game_config.mode,
            "final_scores": final_scores,
            "donation_summary": donation_summary,
            "adjustment_summary": adjustment_summary
        }

    def _calculate_donation(self, final_scores: List[Dict[str, Any]], game_config: GameConfig) -> Dict[str, Any]:
        """计算捐锅"""
        donation_config = game_config.donation_config.get("type", "不捐")
        donation_details = []
        total_donated = 0

        if donation_config == "不捐":
            pass
        elif donation_config == "赢了捐1点":
            for score in final_scores:
                if score["total_score"] > 0:
                    donated = 1
                    donation_details.append({"player_id": score["player_id"], "donated": donated})
                    total_donated += donated
        elif donation_config == "每赢2点捐1点":
            for score in final_scores:
                if score["total_score"] > 0:
                    donated = score["total_score"] // 2
                    if donated > 0:
                        donation_details.append({"player_id": score["player_id"], "donated": donated})
                        total_donated += donated
        elif donation_config == "每赢3点捐1点":
            for score in final_scores:
                if score["total_score"] > 0:
                    donated = score["total_score"] // 3
                    if donated > 0:
                        donation_details.append({"player_id": score["player_id"], "donated": donated})
                        total_donated += donated
        elif donation_config == "赢了全捐":
            for score in final_scores:
                if score["total_score"] > 0:
                    donated = score["total_score"]
                    donation_details.append({"player_id": score["player_id"], "donated": donated})
                    total_donated += donated

        return {
            "total_donated": total_donated,
            "donation_details": donation_details
        }

    def _calculate_score_adjustment(self, final_scores: List[Dict[str, Any]], game_config: GameConfig) -> Dict[
        str, Any]:
        """计算让分调整"""
        adjustment_config = game_config.score_adjustment_config
        mode = adjustment_config.get("mode", "单让")
        adjustment_type = adjustment_config.get("adjustment_type", "实让")
        points = adjustment_config.get("points", 0)

        if points == 0:
            return {
                "mode": mode,
                "type": adjustment_type,
                "points": points,
                "applied": False
            }

        # 计算分数差距
        scores = [s["after_donation"] for s in final_scores]
        max_score = max(scores)
        min_score = min(scores)
        score_diff = max_score - min_score

        applied = False

        if mode == "单让":
            if adjustment_type == "实让":
                # 实让：直接调整分数
                if score_diff > points:
                    # 找到最低分选手，给予让分
                    for score in final_scores:
                        if score["after_donation"] == min_score:
                            score["after_adjustment"] = score["after_donation"] + points
                            applied = True
                            break
            elif adjustment_type == "虚让":
                # 虚让：如果差距小于等于让分，算平局
                if score_diff <= points:
                    # 所有人分数设为0（平局）
                    for score in final_scores:
                        score["after_adjustment"] = 0
                    applied = True
        elif mode == "互虚":
            # 互虚：双向虚让
            if score_diff <= points:
                for score in final_scores:
                    score["after_adjustment"] = 0
                applied = True

        return {
            "mode": mode,
            "type": adjustment_type,
            "points": points,
            "applied": applied
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
def calculate_lasi_score(game_data: LaSiGameData) -> Dict[str, Any]:
    """拉丝三点计分接口函数"""
    logger.info("------------------拉丝三点计分开始--------------------")

    try:
        api = LaSiThreePointAPI()
        result = api.calculate_lasi_score(game_data)

        if result["success"]:
            logger.info("拉丝三点计分成功完成")
        else:
            logger.error(f"拉丝三点计分失败: {result['message']}")

        return result

    except Exception as e:
        logger.error(f"拉丝三点计分接口异常: {str(e)}")
        return {
            "success": False,
            "error_code": "INTERFACE_ERROR",
            "message": f"接口调用异常: {str(e)}",
            "data": None
        }
import json
import copy
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from utils.nlp_logging import CustomLogger

"""
拉丝三点高尔夫赌球游戏简化版计分接口
固化大部分参数，简化使用复杂度

固化规则：
- 基础分: 1分
- PK项目: 3项固定（最好成绩PK、最差成绩PK、总杆数相加PK）
- 取消: 双杀奖励、包赔功能、捐锅功能、高手不见面
- 鸟鹰奖励: 合并PK赢了才奖励，鸟×2、鹰×5、HIO×10
- 平洞规则: 得分差为0，平洞翻倍（不算鸟鹰），par收1/鸟收2/鹰收5

"""

logger = CustomLogger(name="DaoTest 3-Point Las Vegas Simple API", write_to_file=True)

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

# 简化版鸟鹰额外奖励（固化）
SIMPLE_BIRD_EAGLE_REWARDS = {
    -3: 10,  # 信天翁/HIO: ×10
    -2: 5,  # 老鹰: ×5
    -1: 2,  # 小鸟: ×2
}


# 简化版数据模型
class SimpleGameConfig(BaseModel):
    mode: str = Field(..., description="游戏模式: 固拉 | 乱拉")

    # 固拉模式队伍配置
    fixed_teams: Optional[List[Dict[str, Any]]] = None

    # 让杆设置
    handicap_settings: Dict[str, Dict[str, float]] = Field(default_factory=dict)

    # 让杆限制条件
    handicap_restrictions: Dict[str, bool] = Field(default_factory=lambda: {
        "only_total_pk": False,
        "no_leader": False,
        "no_par_bird_eagle": False
    })


class SimplePlayer(BaseModel):
    id: str
    name: str


class SimpleScore(BaseModel):
    player_id: str
    raw_strokes: int


class SimpleHole(BaseModel):
    hole_number: int
    par: int
    hole_type: str = Field(..., description="三杆洞 | 四杆洞 | 五杆洞")
    tee_order: List[str]
    scores: List[SimpleScore]


class SimpleLaSiGameData(BaseModel):
    game_config: SimpleGameConfig
    players: List[SimplePlayer]
    holes: List[SimpleHole]


class LaSiThreePointSimpleAPI:
    def __init__(self):
        self.logger = logger

    def calculate_simple_lasi_score(self, game_data: SimpleLaSiGameData) -> Dict[str, Any]:
        """简化版拉丝三点计分主函数"""
        try:
            self.logger.info("=== 开始简化版拉丝三点计分 ===")

            # 数据验证
            if not self._validate_simple_game_data(game_data):
                return self._error_response("数据验证失败")

            # 初始化计分状态
            score_state = self._initialize_score_state(game_data)

            # 逐洞处理
            hole_details = []
            for hole in game_data.holes:
                hole_result = self._process_simple_hole(hole, game_data, score_state)
                hole_details.append(hole_result)

                # 更新累积状态
                self._update_score_state(score_state, hole_result)

            # 最终结算（简化版无需捐锅和让分）
            final_summary = self._calculate_simple_final_summary(game_data, score_state)

            result = {
                "success": True,
                "data": {
                    "game_summary": final_summary,
                    "hole_details": hole_details,
                    "calculation_details": self._get_fixed_rules_info()
                },
                "message": "简化版计分成功"
            }

            self.logger.info("=== 简化版拉丝三点计分完成 ===")
            return result

        except Exception as e:
            self.logger.error(f"简化版计分过程发生错误: {str(e)}")
            return self._error_response(f"简化版计分失败: {str(e)}")

    def _validate_simple_game_data(self, game_data: SimpleLaSiGameData) -> bool:
        """验证简化版游戏数据"""
        try:
            # 验证玩家数量
            if len(game_data.players) != 4:
                self.logger.error(f"玩家人数错误: {len(game_data.players)}")
                return False

            # 验证固拉模式的队伍配置
            if game_data.game_config.mode == "固拉":
                if not game_data.game_config.fixed_teams or len(game_data.game_config.fixed_teams) != 2:
                    self.logger.error("固拉模式缺少队伍配置")
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
            self.logger.error(f"简化版数据验证异常: {str(e)}")
            return False

    def _initialize_score_state(self, game_data: SimpleLaSiGameData) -> Dict[str, Any]:
        """初始化计分状态"""
        player_ids = [p.id for p in game_data.players]

        return {
            "player_total_scores": {pid: 0 for pid in player_ids},
            "tie_accumulated_score": 0,
            "tie_count": 0,
            "current_tee_order": []
        }

    def _process_simple_hole(self, hole: SimpleHole, game_data: SimpleLaSiGameData,
                             score_state: Dict[str, Any]) -> Dict[str, Any]:
        """处理单洞计分 - 简化版"""
        self.logger.info(f"开始处理第{hole.hole_number}洞（简化版）")

        try:
            # 1. 确定队伍配对
            teams = self._determine_simple_teams(hole, game_data)

            # 2. 计算净杆数（应用让杆）
            net_scores = self._calculate_simple_net_scores(hole, game_data, teams)

            # 3. 执行固定的三项PK
            pk_results = self._execute_simple_pk_comparisons(teams)

            # 4. 判定鸟鹰奖励（简化版固定规则）
            bird_eagle_rewards = self._calculate_simple_bird_eagle_rewards(
                hole, teams, pk_results)

            # 5. 处理平洞逻辑（简化版规则）
            tie_status = self._handle_simple_tie_logic(pk_results, score_state, teams, hole.par)

            # 6. 计算最终洞次得分
            hole_scores = self._calculate_simple_hole_scores(
                teams, pk_results, bird_eagle_rewards, tie_status)

            # 7. 确定下洞开球顺序
            next_tee_order = self._determine_next_tee_order(teams, net_scores)
            score_state["current_tee_order"] = next_tee_order

            return {
                "hole_number": hole.hole_number,
                "par": hole.par,
                "hole_type": hole.hole_type,
                "teams": teams,
                "pk_results": pk_results,
                "special_rewards": {
                    "bird_eagle": bird_eagle_rewards
                },
                "tie_status": tie_status,
                "hole_scores": hole_scores,
                "next_tee_order": next_tee_order
            }

        except Exception as e:
            self.logger.error(f"处理第{hole.hole_number}洞时出错: {str(e)}")
            raise

    def _determine_simple_teams(self, hole: SimpleHole, game_data: SimpleLaSiGameData) -> List[Dict[str, Any]]:
        """确定队伍配对 - 简化版"""
        if game_data.game_config.mode == "固拉":
            # 固拉模式：使用固定队伍
            teams = []
            for fixed_team in game_data.game_config.fixed_teams:
                team_data = {
                    "team_id": fixed_team["team_id"],
                    "team_players": fixed_team["players"],
                    "raw_scores": [],
                    "handicaps": [],
                    "net_scores": []
                }
                teams.append(team_data)
        else:
            # 乱拉模式：默认配对（取消高手不见面）
            tee_order = hole.tee_order
            teams = [
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

        # 填充杆数数据
        score_map = {score.player_id: score.raw_strokes for score in hole.scores}

        for team in teams:
            for player_id in team["team_players"]:
                raw_score = score_map.get(player_id, 0)
                team["raw_scores"].append(raw_score)

        return teams

    def _calculate_simple_net_scores(self, hole: SimpleHole, game_data: SimpleLaSiGameData,
                                     teams: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算净杆数 - 简化版"""
        net_scores = {}

        for team in teams:
            for i, player_id in enumerate(team["team_players"]):
                raw_score = team["raw_scores"][i]

                # 获取让杆数
                handicap = self._get_player_handicap(
                    player_id, hole.hole_type, game_data.game_config.handicap_settings)

                # 检查让杆限制条件
                effective_handicap = self._apply_simple_handicap_restrictions(
                    player_id, handicap, raw_score, hole.par, teams, game_data.game_config)

                net_score = raw_score - effective_handicap

                team["handicaps"].append(effective_handicap)
                team["net_scores"].append(net_score)
                net_scores[player_id] = net_score

        # 计算队伍统计（固定为相加模式）
        for team in teams:
            team["best_score"] = min(team["net_scores"])
            team["worst_score"] = max(team["net_scores"])
            team["total_score"] = sum(team["net_scores"])  # 固定相加

        return net_scores

    def _get_player_handicap(self, player_id: str, hole_type: str,
                             handicap_settings: Dict[str, Dict[str, float]]) -> float:
        """获取选手让杆数"""
        if player_id in handicap_settings:
            return handicap_settings[player_id].get(hole_type, 0.0)
        return 0.0

    def _apply_simple_handicap_restrictions(self, player_id: str, handicap: float, raw_score: int,
                                            par: int, teams: List[Dict[str, Any]],
                                            game_config: SimpleGameConfig) -> float:
        """应用让杆限制条件 - 简化版"""
        restrictions = game_config.handicap_restrictions

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

    def _execute_simple_pk_comparisons(self, teams: List[Dict[str, Any]]) -> Dict[str, Any]:
        """执行简化版三项PK比较"""
        team1, team2 = teams[0], teams[1]
        base_score = 1  # 固定基础分

        # 最好成绩PK
        best_pk = self._compare_scores(team1["best_score"], team2["best_score"], base_score)

        # 最差成绩PK
        worst_pk = self._compare_scores(team1["worst_score"], team2["worst_score"], base_score)

        # 总分PK（固定相加模式）
        total_pk = self._compare_scores(team1["total_score"], team2["total_score"], base_score)
        total_pk["combination_mode"] = "双方总杆相加PK"

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

    def _calculate_simple_bird_eagle_rewards(self, hole: SimpleHole, teams: List[Dict[str, Any]],
                                             pk_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """计算鸟鹰奖励 - 简化版固定规则"""
        rewards = []

        for team in teams:
            for i, player_id in enumerate(team["team_players"]):
                raw_score = team["raw_scores"][i]
                score_to_par = raw_score - hole.par

                # 获取基础分数（基于原始杆数）
                base_score = BIRD_EAGLE_SCORES.get(score_to_par, 0)
                if score_to_par > 4:
                    base_score = BIRD_EAGLE_SCORES[4]  # 四柏忌及以上

                # 获取简化版额外奖励（固定规则）
                extra_reward = SIMPLE_BIRD_EAGLE_REWARDS.get(score_to_par, 0)

                # 检查是否符合奖励条件（固定为合并PK赢了才奖励）
                eligible = self._check_simple_bird_eagle_eligibility(team["team_id"], pk_results)

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

    def _check_simple_bird_eagle_eligibility(self, team_id: int, pk_results: Dict[str, Any]) -> bool:
        """检查鸟鹰奖励资格 - 简化版固定条件"""
        # 计算该队伍的总得分
        team_total = 0
        for pk_name, pk_result in pk_results.items():
            if pk_result["winner_team"] == team_id:
                team_total += pk_result["score_diff"]
            elif pk_result["winner_team"] != 0:  # 不是平局
                team_total -= pk_result["score_diff"]

        return team_total > 0

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

    def _handle_simple_tie_logic(self, pk_results: Dict[str, Any], score_state: Dict[str, Any],
                                 teams: List[Dict[str, Any]], par: int) -> Dict[str, Any]:
        """处理平洞逻辑 - 简化版固定规则"""
        # 检查是否为平洞（固定定义：得分差为0）
        is_tie = self._check_simple_tie_condition(pk_results)

        if is_tie:
            # 平洞：累积分数（平洞翻倍，不算鸟鹰奖励）
            tie_score = 1  # 每次平洞固定1分
            score_state["tie_accumulated_score"] += tie_score
            score_state["tie_count"] += 1

            self.logger.info(
                f"平洞发生，累积分数: {score_state['tie_accumulated_score']}, 累积次数: {score_state['tie_count']}")

            return {
                "is_tie": True,
                "accumulated_score": score_state["tie_accumulated_score"],
                "tie_count": score_state["tie_count"],
                "collected_score": 0
            }
        else:
            # 不是平洞，检查是否需要收取累积分数
            collected_score = 0
            if score_state["tie_accumulated_score"] > 0:
                collected_score = self._collect_simple_tie_score(pk_results, score_state, teams, par)

            return {
                "is_tie": False,
                "accumulated_score": score_state["tie_accumulated_score"],
                "tie_count": score_state["tie_count"],
                "collected_score": collected_score
            }

    def _check_simple_tie_condition(self, pk_results: Dict[str, Any]) -> bool:
        """检查平洞条件 - 简化版固定为得分差为0"""
        total_diff = 0
        for pk_result in pk_results.values():
            if pk_result["winner_team"] == 1:
                total_diff += pk_result["score_diff"]
            elif pk_result["winner_team"] == 2:
                total_diff -= pk_result["score_diff"]

        return total_diff == 0

    def _collect_simple_tie_score(self, pk_results: Dict[str, Any], score_state: Dict[str, Any],
                                  teams: List[Dict[str, Any]], par: int) -> int:
        """收取平洞分数 - 简化版固定规则"""
        if score_state["tie_count"] == 0:
            return 0

        # 确定获胜队伍
        total_winner_score = 0
        for pk_result in pk_results.values():
            if pk_result["winner_team"] == 1:
                total_winner_score += pk_result["score_diff"]
            elif pk_result["winner_team"] == 2:
                total_winner_score -= pk_result["score_diff"]

        if total_winner_score == 0:
            return 0  # 平局不收取

        winner_team_id = 1 if total_winner_score > 0 else 2

        # 检查收取限制条件：获胜队伍最好成绩都是柏忌或更差就不能收取
        winner_team = teams[winner_team_id - 1]
        if winner_team["best_score"] > par:
            self.logger.info(f"获胜队伍最好成绩是柏忌或更差({winner_team['best_score']} > {par})，不能收取平洞分数")
            return 0

        # 确定收取规则
        base_win_score = abs(total_winner_score)  # 本洞赢了几分
        collect_count = base_win_score  # 基础收取：本洞赢几分，收取几个平洞分数

        # 检查是否有人打鸟或鹰，调整收取个数
        has_eagle = False
        has_birdie = False
        for team in teams:
            if team["team_id"] == winner_team_id:
                for i, player_id in enumerate(team["team_players"]):
                    raw_score = team["raw_scores"][i]
                    score_to_par = raw_score - par

                    if score_to_par == -2:  # 打鹰
                        has_eagle = True
                        break
                    elif score_to_par == -1:  # 打鸟
                        has_birdie = True

        # 根据鸟鹰情况调整收取个数
        if has_eagle:
            collect_count = 5  # 打鹰：收取5个平洞分数
        elif has_birdie:
            collect_count = 2  # 打鸟：收取2个平洞分数

        # 计算实际收取分数（不能超过现有平洞次数）
        available_ties = score_state["tie_count"]
        actual_collect_count = min(collect_count, available_ties)

        # 每个平洞1分，收取actual_collect_count个
        collected_score = actual_collect_count * 1

        # 更新累积状态
        score_state["tie_count"] -= actual_collect_count
        score_state["tie_accumulated_score"] = score_state["tie_count"] * 1  # 剩余平洞分数

        self.logger.info(
            f"收取平洞分数: 获胜队伍{winner_team_id}, 本洞赢{base_win_score}分, 打鸟{has_birdie}, 打鹰{has_eagle}, 收取{actual_collect_count}个平洞({collected_score}分)")

        return collected_score

    def _calculate_simple_hole_scores(self, teams: List[Dict[str, Any]], pk_results: Dict[str, Any],
                                      bird_eagle_rewards: List[Dict[str, Any]],
                                      tie_status: Dict[str, Any]) -> List[Dict[str, Any]]:
        """计算最终洞次得分 - 简化版"""
        hole_scores = []

        # 初始化每个选手的得分
        for team in teams:
            for player_id in team["team_players"]:
                hole_scores.append({
                    "player_id": player_id,
                    "base_score": 0,
                    "bird_eagle_score": 0,
                    "extra_reward": 0,
                    "tie_score": 0,
                    "final_score": 0
                })

        # 计算基础PK得分
        self._apply_simple_base_pk_scores(hole_scores, teams, pk_results)

        # 应用鸟鹰分数和奖励
        self._apply_simple_bird_eagle_scores(hole_scores, bird_eagle_rewards)

        # 应用平洞分数
        self._apply_simple_tie_scores(hole_scores, teams, tie_status, pk_results)

        # 计算最终分数
        for score in hole_scores:
            score["final_score"] = (
                    score["base_score"] +
                    score["bird_eagle_score"] +
                    score["extra_reward"] +
                    score["tie_score"]
            )

        return hole_scores

    def _apply_simple_base_pk_scores(self, hole_scores: List[Dict[str, Any]], teams: List[Dict[str, Any]],
                                     pk_results: Dict[str, Any]):
        """应用基础PK得分 - 简化版"""
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

    def _apply_simple_bird_eagle_scores(self, hole_scores: List[Dict[str, Any]],
                                        bird_eagle_rewards: List[Dict[str, Any]]):
        """应用鸟鹰分数和奖励 - 简化版"""
        for reward in bird_eagle_rewards:
            player_id = reward["player_id"]
            for score in hole_scores:
                if score["player_id"] == player_id:
                    if reward["applied"]:
                        score["bird_eagle_score"] = reward["base_score"]
                        score["extra_reward"] = reward["extra_reward"]
                    break

    def _apply_simple_tie_scores(self, hole_scores: List[Dict[str, Any]], teams: List[Dict[str, Any]],
                                 tie_status: Dict[str, Any], pk_results: Dict[str, Any]):
        """应用平洞分数 - 简化版"""
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

    def _determine_next_tee_order(self, teams: List[Dict[str, Any]], net_scores: Dict[str, float]) -> List[str]:
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

    def _calculate_simple_final_summary(self, game_data: SimpleLaSiGameData,
                                        score_state: Dict[str, Any]) -> Dict[str, Any]:
        """计算最终总结 - 简化版"""
        # 基础总分（简化版无需捐锅和让分）
        final_scores = []
        for player in game_data.players:
            total_score = score_state["player_total_scores"][player.id]
            final_scores.append({
                "player_id": player.id,
                "player_name": player.name,
                "total_score": total_score
            })

        return {
            "total_holes": len(game_data.holes),
            "game_mode": game_data.game_config.mode,
            "final_scores": final_scores
        }

    def _get_fixed_rules_info(self) -> Dict[str, Any]:
        """获取固定规则信息"""
        return {
            "fixed_rules": {
                "base_score": 1,
                "pk_items": ["最好成绩PK", "最差成绩PK", "总杆数相加PK"],
                "bird_eagle_condition": "合并PK项赢了才奖励",
                "bird_eagle_rewards": {
                    "小鸟": "×2",
                    "老鹰": "×5",
                    "信天翁": "×10"
                },
                "tie_definition": "得分差为0",
                "tie_scoring": "平洞翻倍（不算鸟鹰奖励）",
                "tie_collect_rules": {
                    "基础": "赢几分收几分",
                    "打鸟": "收2倍平洞分数",
                    "打鹰": "收5倍平洞分数",
                    "限制": "获胜队伍最好成绩都是柏忌或更差时不能收取"
                }
            },
            "removed_features": [
                "双杀奖励", "包赔功能", "捐锅功能", "高手不见面", "让分调整"
            ]
        }

    def _error_response(self, message: str) -> Dict[str, Any]:
        """返回错误响应"""
        return {
            "success": False,
            "error_code": "SIMPLE_CALCULATION_ERROR",
            "message": message,
            "data": None
        }


# 简化版接口函数
def calculate_simple_lasi_score(game_data: SimpleLaSiGameData) -> Dict[str, Any]:
    """简化版拉丝三点计分接口函数"""
    logger.info("------------------简化版拉丝三点计分开始--------------------")

    try:
        api = LaSiThreePointSimpleAPI()
        result = api.calculate_simple_lasi_score(game_data)

        if result["success"]:
            logger.info("简化版拉丝三点计分成功完成")
        else:
            logger.error(f"简化版拉丝三点计分失败: {result['message']}")

        return result

    except Exception as e:
        logger.error(f"简化版拉丝三点计分接口异常: {str(e)}")
        return {
            "success": False,
            "error_code": "SIMPLE_INTERFACE_ERROR",
            "message": f"简化版接口调用异常: {str(e)}",
            "data": None
        }

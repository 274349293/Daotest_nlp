import json
import copy
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
import logging
from datetime import datetime

"""
拉丝三点高尔夫赌球游戏计分接口
完整实现拉丝三点游戏计分逻辑，包括：
- 固拉/乱拉两种模式
- 高手不见面配置
- 组合PK相加/相乘模式
- 三项比较计分（最好/最差/组合）
- 鸟鹰基础分数和额外奖励系统
- 双杀奖励机制
- 平洞累积和收取规则
- 包赔规则判定和分数重分配
- 复杂让杆规则和条件限制
- 捐锅和让分最终结算
"""

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lasi_scoring.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("LaSiThreePoint")

# 鸟鹰基础分数系统（基于原始杆数与标准杆的差值）
BIRD_EAGLE_SCORES = {
    -3: 32,  # 信天翁 (Albatross)
    -2: 16,  # 老鹰 (Eagle)
    -1: 8,  # 小鸟 (Birdie)
    0: 4,  # 标准杆 (Par)
    1: 2,  # 柏忌 (Bogey)
    2: 1,  # 双柏忌 (Double Bogey)
    3: 0,  # 三柏忌 (Triple Bogey)
    4: -1  # 四柏忌及以上 (Quadruple Bogey+)
}


# 数据模型定义
class FixedTeam(BaseModel):
    """固定队伍配置"""
    team_id: int
    players: List[str]


class CombinationPKConfig(BaseModel):
    """组合PK配置"""
    mode: str = Field(default="双方总杆相加PK", description="双方总杆相加PK | 双方总杆相乘PK")


class ExpertSeparationConfig(BaseModel):
    """高手不见面配置"""
    enabled: bool = Field(default=False, description="是否启用高手不见面")
    expert_players: List[str] = Field(default_factory=list, description="高手玩家ID列表，必须2人")


class DoubleKillConfig(BaseModel):
    """双杀配置"""
    type: str = Field(default="不奖励", description="不奖励|奖励1分|奖励2分|奖励3分|翻倍奖励")


class BirdEagleConfig(BaseModel):
    """鸟鹰配置"""
    condition: str = Field(default="合并pk赢了才奖励", description="鸟鹰奖励条件")
    extra_reward: str = Field(default="不奖励", description="额外奖励类型")


class TieConfig(BaseModel):
    """平洞配置"""
    definition: str = Field(default="得分差为0", description="平洞定义")
    scoring: str = Field(default="平洞跳过(无肉)", description="平洞计分规则")
    collect_rule: str = Field(default="赢了全收掉", description="平洞收取规则")


class CompensationConditions(BaseModel):
    """包赔条件"""
    double_par: bool = Field(default=False, description="双par及以上")
    plus_three: bool = Field(default=False, description="+3及以上")
    diff_three: bool = Field(default=False, description="相差3杆及以上")


class CompensationConfig(BaseModel):
    """包赔配置"""
    scope: str = Field(default="不包赔", description="不包赔|包本洞所有分|包含平洞的所有分")
    conditions: CompensationConditions = Field(default_factory=CompensationConditions)


class DonationConfig(BaseModel):
    """捐锅配置"""
    type: str = Field(default="不捐", description="不捐|赢了捐1点|每赢2点捐1点|每赢3点捐1点|赢了全捐")


class HandicapRestrictions(BaseModel):
    """让杆限制条件"""
    only_combination_pk: bool = Field(default=False, description="仅组合PK让杆")
    no_leader: bool = Field(default=False, description="打头不让杆")
    no_par_bird_eagle: bool = Field(default=False, description="Par/鸟/鹰不让杆")


class HandicapConfig(BaseModel):
    """让杆配置"""
    restrictions: HandicapRestrictions = Field(default_factory=HandicapRestrictions)


class ScoreAdjustmentConfig(BaseModel):
    """让分配置"""
    mode: str = Field(default="单让", description="单让|互虚")
    adjustment_type: str = Field(default="实让", description="实让|虚让")
    points: int = Field(default=0, description="让分数")


class GameConfig(BaseModel):
    """游戏配置"""
    mode: str = Field(..., description="游戏模式: 固拉 | 乱拉")
    base_score: int = Field(default=1, description="基础分")
    fixed_teams: Optional[List[FixedTeam]] = Field(default=None, description="固定队伍配置，仅固拉模式使用")
    combination_pk_config: CombinationPKConfig = Field(default_factory=CombinationPKConfig)
    expert_separation_config: ExpertSeparationConfig = Field(default_factory=ExpertSeparationConfig)
    double_kill_config: DoubleKillConfig = Field(default_factory=DoubleKillConfig)
    bird_eagle_config: BirdEagleConfig = Field(default_factory=BirdEagleConfig)
    tie_config: TieConfig = Field(default_factory=TieConfig)
    compensation_config: CompensationConfig = Field(default_factory=CompensationConfig)
    donation_config: DonationConfig = Field(default_factory=DonationConfig)
    handicap_config: HandicapConfig = Field(default_factory=HandicapConfig)
    score_adjustment_config: ScoreAdjustmentConfig = Field(default_factory=ScoreAdjustmentConfig)
    handicap_settings: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="让杆设置")


class Player(BaseModel):
    """选手信息"""
    id: str
    name: str


class Score(BaseModel):
    """单洞得分"""
    player_id: str
    raw_strokes: int


class Hole(BaseModel):
    """洞次信息"""
    hole_number: int
    par: int
    hole_type: str = Field(..., description="三杆洞 | 四杆洞 | 五杆洞")
    tee_order: List[str]
    scores: List[Score]


class LaSiGameData(BaseModel):
    """拉丝游戏数据"""
    game_config: GameConfig
    players: List[Player]
    holes: List[Hole]


class LaSiThreePointScoring:
    """拉丝三点计分系统"""

    def __init__(self):
        self.logger = logger

    def _validate_config_parameters(self, game_config: GameConfig) -> Tuple[bool, List[str]]:
        """校验游戏配置参数，返回(是否通过, 错误信息列表)"""
        errors = []

        # 定义所有合法参数值
        VALID_MODES = ["固拉", "乱拉"]

        VALID_COMBINATION_PK_MODES = ["双方总杆相加PK", "双方总杆相乘PK"]

        VALID_DOUBLE_KILL_TYPES = ["不奖励", "奖励1分", "奖励2分", "奖励3分", "翻倍奖励"]

        VALID_BIRD_EAGLE_CONDITIONS = ["合并pk赢了才奖励"]

        VALID_BIRD_EAGLE_EXTRA_REWARDS = [
            "不奖励",
            "鸟+1/鹰+4/HIO+9", "鸟+1/鹰+4/HIO+8", "鸟+1/鹰+5/HIO+10", "鸟+1/鹰+10/HIO+20",
            "鸟*2/鹰*4/HIO*8", "鸟*2/鹰*5/HIO*10"
        ]

        VALID_TIE_DEFINITIONS = ["得分差为0"]

        VALID_TIE_SCORING = [
            "平洞跳过(无肉)", "平洞算1点", "平洞算2点", "平洞算3点", "平洞算4点",
            "平洞翻倍(不算鸟鹰奖)", "平洞翻倍(算鸟鹰奖)", "平洞连续翻番"
        ]

        VALID_TIE_COLLECT_RULES = [
            "赢了全收掉",
            "Par收1/鸟收2/鹰收4", "Par收1/鸟收2/鹰收5", "Par收1/鸟收2/鹰全收",
            "赢了收1洞", "赢了收2洞", "赢了收3洞", "赢了收4洞", "赢了收5洞"
        ]

        VALID_COMPENSATION_SCOPES = ["不包赔", "包本洞所有分", "包含平洞的所有分"]

        VALID_DONATION_TYPES = ["不捐", "赢了捐1点", "每赢2点捐1点", "每赢3点捐1点", "赢了全捐"]

        VALID_ADJUSTMENT_MODES = ["单让", "互虚"]

        VALID_ADJUSTMENT_TYPES = ["实让", "虚让"]

        VALID_HOLE_TYPES = ["三杆洞", "四杆洞", "五杆洞"]

        # 开始校验
        self.logger.info("开始参数校验...")

        # 1. 游戏模式
        if game_config.mode not in VALID_MODES:
            errors.append(f"无效的游戏模式: '{game_config.mode}', 有效值: {VALID_MODES}")

        # 2. 组合PK配置
        combination_mode = game_config.combination_pk_config.mode
        if combination_mode not in VALID_COMBINATION_PK_MODES:
            errors.append(f"无效的组合PK模式: '{combination_mode}', 有效值: {VALID_COMBINATION_PK_MODES}")

        # 3. 双杀配置
        double_kill_type = game_config.double_kill_config.type
        if double_kill_type not in VALID_DOUBLE_KILL_TYPES:
            errors.append(f"无效的双杀类型: '{double_kill_type}', 有效值: {VALID_DOUBLE_KILL_TYPES}")

        # 4. 鸟鹰配置
        bird_eagle_condition = game_config.bird_eagle_config.condition
        if bird_eagle_condition not in VALID_BIRD_EAGLE_CONDITIONS:
            errors.append(f"无效的鸟鹰条件: '{bird_eagle_condition}', 有效值: {VALID_BIRD_EAGLE_CONDITIONS}")

        bird_eagle_extra = game_config.bird_eagle_config.extra_reward
        if bird_eagle_extra not in VALID_BIRD_EAGLE_EXTRA_REWARDS:
            errors.append(f"无效的鸟鹰额外奖励: '{bird_eagle_extra}', 有效值: {VALID_BIRD_EAGLE_EXTRA_REWARDS}")

        # 5. 平洞配置
        tie_definition = game_config.tie_config.definition
        if tie_definition not in VALID_TIE_DEFINITIONS:
            errors.append(f"无效的平洞定义: '{tie_definition}', 有效值: {VALID_TIE_DEFINITIONS}")

        tie_scoring = game_config.tie_config.scoring
        if tie_scoring not in VALID_TIE_SCORING:
            errors.append(f"无效的平洞计分规则: '{tie_scoring}', 有效值: {VALID_TIE_SCORING}")

        tie_collect = game_config.tie_config.collect_rule
        if tie_collect not in VALID_TIE_COLLECT_RULES:
            errors.append(f"无效的平洞收取规则: '{tie_collect}', 有效值: {VALID_TIE_COLLECT_RULES}")

        # 6. 包赔配置
        compensation_scope = game_config.compensation_config.scope
        if compensation_scope not in VALID_COMPENSATION_SCOPES:
            errors.append(f"无效的包赔范围: '{compensation_scope}', 有效值: {VALID_COMPENSATION_SCOPES}")

        # 7. 捐锅配置
        donation_type = game_config.donation_config.type
        if donation_type not in VALID_DONATION_TYPES:
            errors.append(f"无效的捐锅类型: '{donation_type}', 有效值: {VALID_DONATION_TYPES}")

        # 8. 让分配置（仅固拉模式）
        if game_config.mode == "固拉":
            adjustment_mode = game_config.score_adjustment_config.mode
            if adjustment_mode not in VALID_ADJUSTMENT_MODES:
                errors.append(f"无效的让分模式: '{adjustment_mode}', 有效值: {VALID_ADJUSTMENT_MODES}")

            adjustment_type = game_config.score_adjustment_config.adjustment_type
            if adjustment_type not in VALID_ADJUSTMENT_TYPES:
                errors.append(f"无效的让分类型: '{adjustment_type}', 有效值: {VALID_ADJUSTMENT_TYPES}")

        # 9. 让杆配置中的洞次类型
        for player_id, handicap_settings in game_config.handicap_settings.items():
            for hole_type, handicap in handicap_settings.items():
                if hole_type not in VALID_HOLE_TYPES:
                    errors.append(f"选手 '{player_id}' 无效的洞次类型: '{hole_type}', 有效值: {VALID_HOLE_TYPES}")

        # 输出校验结果
        if errors:
            self.logger.error("参数校验失败:")
            for error in errors:
                self.logger.error(f"  - {error}")
            return False, errors
        else:
            self.logger.info("参数校验通过")
            return True, []

    def calculate_score(self, game_data: LaSiGameData) -> Dict[str, Any]:
        """主计分函数"""
        self.logger.info("=" * 80)
        self.logger.info("开始拉丝三点高尔夫赌球游戏计分")
        self.logger.info("=" * 80)

        try:
            # 0. 参数校验 - 新增
            is_valid, validation_errors = self._validate_config_parameters(game_data.game_config)
            if not is_valid:
                return self._create_error_response(f"参数校验失败: {'; '.join(validation_errors)}")
            # 1. 数据验证
            if not self._validate_game_data(game_data):
                return self._create_error_response("数据验证失败")

            # 2. 初始化计分状态
            scoring_state = self._initialize_scoring_state(game_data)

            # 3. 逐洞处理
            hole_details = []
            for hole in game_data.holes:
                self.logger.info(f"\n{'=' * 60}")
                self.logger.info(f"开始处理第{hole.hole_number}洞 (Par {hole.par}, {hole.hole_type})")
                self.logger.info(f"{'=' * 60}")

                hole_result = self._process_hole(hole, game_data, scoring_state)
                hole_details.append(hole_result)

                # 更新累积状态
                self._update_scoring_state(scoring_state, hole_result)

                self.logger.info(f"第{hole.hole_number}洞处理完成")

            # 4. 最终结算
            final_summary = self._calculate_final_summary(game_data, scoring_state, hole_details)

            result = {
                "success": True,
                "data": {
                    "game_summary": final_summary,
                    "hole_details": hole_details
                },
                "message": "计分成功"
            }

            self.logger.info("=" * 80)
            self.logger.info("拉丝三点计分完成")
            self.logger.info("=" * 80)

            return result

        except Exception as e:
            self.logger.error(f"计分过程发生错误: {str(e)}", exc_info=True)
            return self._create_error_response(f"计分失败: {str(e)}")

    def _validate_game_data(self, game_data: LaSiGameData) -> bool:
        """验证游戏数据"""
        self.logger.info("开始验证游戏数据...")

        # 验证玩家数量
        if len(game_data.players) != 4:
            self.logger.error(f"玩家人数错误: 需要4人，当前{len(game_data.players)}人")
            return False

        player_names = [p.name for p in game_data.players]
        self.logger.info(f"参赛选手: {', '.join(player_names)}")

        # 验证模式特定配置
        if game_data.game_config.mode == "固拉":
            if not game_data.game_config.fixed_teams or len(game_data.game_config.fixed_teams) != 2:
                self.logger.error("固拉模式缺少正确的队伍配置")
                return False

            self.logger.info("固拉模式配置验证通过")
            for team in game_data.game_config.fixed_teams:
                team_names = [next(p.name for p in game_data.players if p.id == pid) for pid in team.players]
                self.logger.info(f"队伍{team.team_id}: {', '.join(team_names)}")

        elif game_data.game_config.mode == "乱拉":
            expert_config = game_data.game_config.expert_separation_config
            if expert_config.enabled:
                if len(expert_config.expert_players) != 2:
                    self.logger.error("高手不见面模式必须选择2名高手")
                    return False

                player_ids = [p.id for p in game_data.players]
                for expert_id in expert_config.expert_players:
                    if expert_id not in player_ids:
                        self.logger.error(f"高手玩家ID不存在: {expert_id}")
                        return False

                expert_names = [next(p.name for p in game_data.players if p.id == eid)
                                for eid in expert_config.expert_players]
                self.logger.info(f"乱拉模式(高手不见面): {', '.join(expert_names)}")
            else:
                self.logger.info("乱拉模式(默认配对)")

        # 验证洞次数据
        if not game_data.holes:
            self.logger.error("缺少洞次数据")
            return False

        for hole in game_data.holes:
            if len(hole.scores) != 4:
                self.logger.error(f"第{hole.hole_number}洞分数数据不完整")
                return False

        self.logger.info(f"总共{len(game_data.holes)}洞比赛")
        self.logger.info("游戏数据验证通过")

        return True

    def _initialize_scoring_state(self, game_data: LaSiGameData) -> Dict[str, Any]:
        """初始化计分状态"""
        player_ids = [p.id for p in game_data.players]

        state = {
            "player_total_scores": {pid: 0 for pid in player_ids},
            "tie_accumulated_score": 0,
            "tie_count": 0,
            "current_tee_order": []
        }

        self.logger.info("计分状态初始化完成")
        return state

    def _process_hole(self, hole: Hole, game_data: LaSiGameData, scoring_state: Dict[str, Any]) -> Dict[str, Any]:
        """处理单洞计分"""

        # 1. 确定队伍配对
        teams = self._determine_teams(hole, game_data)

        # 2. 计算净杆数（应用让杆）
        self._calculate_net_scores(hole, game_data, teams)

        # 3. 执行三项比较PK
        pk_results = self._execute_pk_comparisons(teams, game_data.game_config)

        # 4. 计算特殊奖励
        special_rewards = self._calculate_special_rewards(hole, teams, pk_results, game_data.game_config,
                                                          game_data.players)

        # 5. 处理包赔逻辑
        compensation = self._handle_compensation(hole, teams, game_data.game_config, game_data.players)

        # 6. 处理平洞逻辑
        tie_status = self._handle_tie_logic(pk_results, scoring_state, game_data.game_config, hole, teams)

        # 7. 计算最终洞次得分
        hole_scores = self._calculate_hole_scores(teams, pk_results, special_rewards,
                                                  compensation, tie_status, game_data.game_config, game_data.players)

        # 8. 确定下洞开球顺序
        next_tee_order = self._determine_next_tee_order(hole, teams, game_data.players)
        scoring_state["current_tee_order"] = next_tee_order

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

    def _determine_teams(self, hole: Hole, game_data: LaSiGameData) -> List[Dict[str, Any]]:
        """确定队伍配对"""
        self.logger.info("\n--- 确定队伍配对 ---")

        if game_data.game_config.mode == "固拉":
            # 固拉模式：使用固定队伍
            teams = []
            for fixed_team in game_data.game_config.fixed_teams:
                team_players = fixed_team.players
                team_names = [next(p.name for p in game_data.players if p.id == pid) for pid in team_players]

                self.logger.info(f"队伍{fixed_team.team_id}: {', '.join(team_names)}")

                teams.append({
                    "team_id": fixed_team.team_id,
                    "team_players": team_players,
                    "raw_scores": [],
                    "handicaps": [],
                    "net_scores": []
                })
        else:
            # 乱拉模式
            teams = self._determine_luanla_teams(hole, game_data)

        # 填充杆数数据
        score_map = {score.player_id: score.raw_strokes for score in hole.scores}

        for team in teams:
            for player_id in team["team_players"]:
                raw_score = score_map.get(player_id, 0)
                team["raw_scores"].append(raw_score)

        # 打印原始杆数
        self.logger.info("\n原始杆数:")
        for team in teams:
            for i, player_id in enumerate(team["team_players"]):
                player_name = next(p.name for p in game_data.players if p.id == player_id)
                self.logger.info(f"  {player_name}: {team['raw_scores'][i]}杆")

        return teams

    def _determine_luanla_teams(self, hole: Hole, game_data: LaSiGameData) -> List[Dict[str, Any]]:
        """确定乱拉模式的队伍配对"""
        tee_order = hole.tee_order
        expert_config = game_data.game_config.expert_separation_config

        # 打印开球顺序
        tee_names = [next(p.name for p in game_data.players if p.id == pid) for pid in tee_order]
        self.logger.info(f"开球顺序: {', '.join(tee_names)}")

        if expert_config.enabled:
            # 高手不见面模式
            expert_players = expert_config.expert_players

            # 验证高手是否都在开球顺序中
            if not all(expert in tee_order for expert in expert_players):
                self.logger.warning("高手玩家不在开球顺序中，使用默认配对")
                return self._default_luanla_pairing(tee_order, game_data.players)

            # 确保高手分在不同队伍
            expert1, expert2 = expert_players[0], expert_players[1]
            other_players = [p for p in tee_order if p not in expert_players]

            # 将高手分别放在队伍1和队伍2
            team1_players = [expert1, other_players[0]]
            team2_players = [expert2, other_players[1]]

            team1_names = [next(p.name for p in game_data.players if p.id == pid) for pid in team1_players]
            team2_names = [next(p.name for p in game_data.players if p.id == pid) for pid in team2_players]

            self.logger.info(f"队伍1 (高手不见面): {', '.join(team1_names)}")
            self.logger.info(f"队伍2 (高手不见面): {', '.join(team2_names)}")

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
        else:
            # 默认乱拉配对
            teams = self._default_luanla_pairing(tee_order, game_data.players)

        return teams

    def _default_luanla_pairing(self, tee_order: List[str], players: List[Player]) -> List[Dict[str, Any]]:
        """默认乱拉配对模式：第1&第4 vs 第2&第3"""
        team1_players = [tee_order[0], tee_order[3]]  # 第1和第4
        team2_players = [tee_order[1], tee_order[2]]  # 第2和第3

        team1_names = [next(p.name for p in players if p.id == pid) for pid in team1_players]
        team2_names = [next(p.name for p in players if p.id == pid) for pid in team2_players]

        self.logger.info(f"队伍1 (默认配对): {', '.join(team1_names)}")
        self.logger.info(f"队伍2 (默认配对): {', '.join(team2_names)}")

        return [
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

    def _calculate_net_scores(self, hole: Hole, game_data: LaSiGameData, teams: List[Dict[str, Any]]):
        """计算净杆数（应用让杆）"""
        self.logger.info("\n--- 计算净杆数（应用让杆）---")

        for team in teams:
            for i, player_id in enumerate(team["team_players"]):
                player_name = next(p.name for p in game_data.players if p.id == player_id)
                raw_score = team["raw_scores"][i]

                # 获取让杆数
                handicap = self._get_player_handicap(player_id, hole.hole_type,
                                                     game_data.game_config.handicap_settings)

                # 检查让杆限制条件
                effective_handicap = self._apply_handicap_restrictions(
                    player_id, player_name, handicap, raw_score, hole.par, teams, game_data
                )

                # 计算净杆数
                net_score = max(0.5, raw_score - effective_handicap)  # 最小值为0.5

                team["handicaps"].append(effective_handicap)
                team["net_scores"].append(net_score)

                if effective_handicap != 0:
                    self.logger.info(f"  {player_name}: {raw_score}杆 - {effective_handicap}让杆 = {net_score}净杆")
                else:
                    self.logger.info(f"  {player_name}: {raw_score}杆 (无让杆)")

        # 计算队伍统计
        for team in teams:
            team["best_score"] = min(team["net_scores"])
            team["worst_score"] = max(team["net_scores"])

            # 根据组合PK配置计算总分
            combination_mode = game_data.game_config.combination_pk_config.mode
            if combination_mode == "双方总杆相乘PK":
                team["total_score"] = team["net_scores"][0] * team["net_scores"][1]
                self.logger.info(
                    f"队伍{team['team_id']} 组合分数(相乘): {team['net_scores'][0]} × {team['net_scores'][1]} = {team['total_score']}")
            else:  # 默认相加
                team["total_score"] = sum(team["net_scores"])
                self.logger.info(
                    f"队伍{team['team_id']} 组合分数(相加): {team['net_scores'][0]} + {team['net_scores'][1]} = {team['total_score']}")

            self.logger.info(f"队伍{team['team_id']} 最好: {team['best_score']}, 最差: {team['worst_score']}")

    def _get_player_handicap(self, player_id: str, hole_type: str,
                             handicap_settings: Dict[str, Dict[str, float]]) -> float:
        """获取选手让杆数"""
        if player_id in handicap_settings:
            return handicap_settings[player_id].get(hole_type, 0.0)
        return 0.0

    def _apply_handicap_restrictions(self, player_id: str, player_name: str, handicap: float,
                                     raw_score: int, par: int, teams: List[Dict[str, Any]],
                                     game_data: LaSiGameData) -> float:
        """应用让杆限制条件"""
        restrictions = game_data.game_config.handicap_config.restrictions

        # 检查是否为该洞成绩最好的选手（打头不让）
        if restrictions.no_leader:
            all_raw_scores = []
            for team in teams:
                all_raw_scores.extend(team["raw_scores"])

            if raw_score == min(all_raw_scores):
                self.logger.info(f"    {player_name} 打头不让杆 (原让杆: {handicap})")
                return 0.0

        # 检查是否打出Par/鸟/鹰（不让杆）
        if restrictions.no_par_bird_eagle:
            score_to_par = raw_score - par
            if score_to_par <= 0:  # Par或更好
                achievement = "标准杆" if score_to_par == 0 else ("小鸟" if score_to_par == -1 else "老鹰")
                self.logger.info(f"    {player_name} 打出{achievement}不让杆 (原让杆: {handicap})")
                return 0.0

        return handicap

    def _execute_pk_comparisons(self, teams: List[Dict[str, Any]], game_config: GameConfig) -> Dict[str, Any]:
        """执行三项比较PK"""
        self.logger.info("\n--- 三项比较PK ---")

        team1, team2 = teams[0], teams[1]
        base_score = game_config.base_score

        # 最好成绩PK
        best_pk = self._compare_scores("最好成绩PK", team1["best_score"], team2["best_score"], base_score)

        # 最差成绩PK
        worst_pk = self._compare_scores("最差成绩PK", team1["worst_score"], team2["worst_score"], base_score)

        # 组合PK
        combination_mode = game_config.combination_pk_config.mode
        combination_pk = self._compare_scores(f"组合PK({combination_mode})",
                                              team1["total_score"], team2["total_score"], base_score)
        combination_pk["combination_mode"] = combination_mode

        return {
            "best_pk": best_pk,
            "worst_pk": worst_pk,
            "combination_pk": combination_pk
        }

    def _compare_scores(self, pk_name: str, team1_score: float, team2_score: float, base_score: int) -> Dict[str, Any]:
        """比较两队得分"""
        if team1_score < team2_score:
            winner = 1
            diff = base_score
            self.logger.info(f"  {pk_name}: 队伍1胜 ({team1_score} < {team2_score}), 得{base_score}分")
        elif team1_score > team2_score:
            winner = 2
            diff = base_score
            self.logger.info(f"  {pk_name}: 队伍2胜 ({team1_score} > {team2_score}), 得{base_score}分")
        else:
            winner = 0
            diff = 0
            self.logger.info(f"  {pk_name}: 平局 ({team1_score} = {team2_score})")

        return {
            "winner_team": winner,
            "score_diff": diff,
            "team1_score": team1_score,
            "team2_score": team2_score
        }

    def _calculate_special_rewards(self, hole: Hole, teams: List[Dict[str, Any]],
                                   pk_results: Dict[str, Any], game_config: GameConfig,
                                   players: List[Player]) -> Dict[str, Any]:
        """计算特殊奖励"""
        self.logger.info("\n--- 特殊奖励计算 ---")

        # 双杀奖励
        double_kill = self._check_double_kill(teams, game_config)

        # 鸟鹰奖励
        bird_eagle = self._calculate_bird_eagle_rewards(hole, teams, pk_results, game_config, players)

        return {
            "double_kill": double_kill,
            "bird_eagle": bird_eagle
        }

    def _check_double_kill(self, teams: List[Dict[str, Any]], game_config: GameConfig) -> Dict[str, Any]:
        """检查双杀奖励"""
        team1, team2 = teams[0], teams[1]

        # 双杀条件：一队最差成绩比对方最好成绩还要好（基于净杆数）
        team1_double_kill = team1["worst_score"] < team2["best_score"]
        team2_double_kill = team2["worst_score"] < team1["best_score"]

        if team1_double_kill:
            winner_team = 1
            self.logger.info(f"双杀触发！队伍1最差({team1['worst_score']}) < 队伍2最好({team2['best_score']})")
        elif team2_double_kill:
            winner_team = 2
            self.logger.info(f"双杀触发！队伍2最差({team2['worst_score']}) < 队伍1最好({team1['best_score']})")
        else:
            self.logger.info("无双杀")
            return {"triggered": False, "team_id": None, "reward_points": 0}

        # 获取双杀奖励分数
        double_kill_config = game_config.double_kill_config.type
        reward_points = self._parse_double_kill_reward(double_kill_config)

        self.logger.info(f"双杀奖励: {double_kill_config}, 队伍{winner_team}每人获得{reward_points}分")

        return {
            "triggered": True,
            "team_id": winner_team,
            "reward_points": reward_points
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
            return -1  # 特殊标记，表示翻倍
        else:
            return 0

    def _get_team_base_score(self, team_id: int, pk_results: Dict[str, Any]) -> int:
        """获取队伍在三项PK中的基础得分"""
        team_score = 0
        for pk_result in pk_results.values():
            if pk_result["winner_team"] == team_id:
                team_score += pk_result["score_diff"]
            elif pk_result["winner_team"] != 0:  # 不是平局
                team_score -= pk_result["score_diff"]
        return team_score

    def _calculate_bird_eagle_rewards(self, hole: Hole, teams: List[Dict[str, Any]],
                                      pk_results: Dict[str, Any], game_config: GameConfig,
                                      players: List[Player]) -> List[Dict[str, Any]]:
        """计算鸟鹰奖励"""
        rewards = []
        condition = game_config.bird_eagle_config.condition

        self.logger.info("鸟鹰奖励计算:")

        for team in teams:
            # 获取该队伍的PK基础得分
            team_base_score = self._get_team_base_score(team["team_id"], pk_results)

            # 检查队伍是否符合奖励条件
            team_eligible = self._check_bird_eagle_eligibility(team["team_id"], pk_results, condition)

            # 统计队伍内的鸟鹰成就
            team_bird_eagle_multiplier = 1
            team_achievements = []

            for i, player_id in enumerate(team["team_players"]):
                raw_score = team["raw_scores"][i]
                score_to_par = raw_score - hole.par
                achievement_name = self._get_achievement_name(score_to_par)

                # 只计算优于标准杆的成绩
                if score_to_par <= 0:
                    # 获取该成就的倍数
                    _, multiplier = self._get_extra_bird_eagle_reward(score_to_par, game_config)
                    if multiplier > 0:
                        team_bird_eagle_multiplier *= multiplier
                        team_achievements.append({
                            "player_id": player_id,
                            "achievement": achievement_name,
                            "score_to_par": score_to_par,
                            "multiplier": multiplier
                        })

            # 计算队伍鸟鹰总分
            team_bird_eagle_score = 0
            if team_eligible and team_base_score > 0 and team_achievements:
                team_bird_eagle_score = team_base_score * team_bird_eagle_multiplier

                # 打印队伍鸟鹰奖励详情
                achievement_desc = ", ".join([f"{a['achievement']}×{a['multiplier']}" for a in team_achievements])
                self.logger.info(
                    f"队伍{team['team_id']} 鸟鹰奖励: 基础{team_base_score}分 × {team_bird_eagle_multiplier}({achievement_desc}) = {team_bird_eagle_score}分")

            # 为队伍内每个选手创建奖励记录
            for i, player_id in enumerate(team["team_players"]):
                raw_score = team["raw_scores"][i]
                score_to_par = raw_score - hole.par
                achievement_name = self._get_achievement_name(score_to_par)

                # 获取个人成就信息
                personal_multiplier = 0
                personal_extra = 0
                if score_to_par <= 0:
                    personal_extra, personal_multiplier = self._get_extra_bird_eagle_reward(score_to_par, game_config)

                # 个人获得的鸟鹰分数等于队伍鸟鹰分数
                final_bird_eagle_score = team_bird_eagle_score if team_achievements else 0

                player_name = next((p.name for p in players if p.id == player_id), player_id)

                # 打印个人鸟鹰信息
                if achievement_name in ["小鸟", "老鹰", "信天翁"]:
                    if team_eligible and team_base_score > 0:
                        self.logger.info(f"  {player_name}: {achievement_name} - 获得{final_bird_eagle_score}分")
                    else:
                        if team_base_score <= 0:
                            self.logger.info(f"  {player_name}: {achievement_name} - 队伍未获得PK净胜分")
                        else:
                            self.logger.info(f"  {player_name}: {achievement_name} - 不符合奖励条件({condition})")

                rewards.append({
                    "player_id": player_id,
                    "achievement": achievement_name,
                    "raw_score": raw_score,
                    "team_base_score": team_base_score,
                    "base_points": BIRD_EAGLE_SCORES.get(score_to_par, 0),  # 保留原积分表分数用于记录
                    "extra_reward": personal_extra,
                    "multiplier": personal_multiplier,
                    "team_multiplier": team_bird_eagle_multiplier,
                    "final_bird_eagle_score": final_bird_eagle_score,
                    "eligible": team_eligible,
                    "applied": team_eligible and final_bird_eagle_score > 0
                })

        return rewards

    def _get_extra_bird_eagle_reward(self, score_to_par: int, game_config: GameConfig) -> Tuple[int, int]:
        """获取鸟鹰额外奖励，返回(额外分数, 倍数)"""
        extra_reward_config = game_config.bird_eagle_config.extra_reward

        if extra_reward_config == "不奖励":
            return 0, 0

        # 解析固定额外奖励配置
        fixed_reward_map = {
            "鸟+1/鹰+4/HIO+9": {-1: 1, -2: 4, -3: 9},
            "鸟+1/鹰+4/HIO+8": {-1: 1, -2: 4, -3: 8},
            "鸟+1/鹰+5/HIO+10": {-1: 1, -2: 5, -3: 10},
            "鸟+1/鹰+10/HIO+20": {-1: 1, -2: 10, -3: 20}
        }

        # 解析倍数奖励配置
        multiplier_reward_map = {
            "鸟*2/鹰*4/HIO*8": {-1: 2, -2: 4, -3: 8},
            "鸟*2/鹰*5/HIO*10": {-1: 2, -2: 5, -3: 10}
        }

        if extra_reward_config in fixed_reward_map:
            extra_points = fixed_reward_map[extra_reward_config].get(score_to_par, 0)
            return extra_points, 0
        elif extra_reward_config in multiplier_reward_map:
            multiplier = multiplier_reward_map[extra_reward_config].get(score_to_par, 1)
            return 0, multiplier

        return 0, 0

    def _check_bird_eagle_eligibility(self, team_id: int, pk_results: Dict[str, Any], condition: str) -> bool:
        """检查鸟鹰奖励资格"""
        if condition == "合并pk赢了才奖励":
            # 计算该队伍在三项PK中的总净胜分
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

    def _handle_compensation(self, hole: Hole, teams: List[Dict[str, Any]],
                             game_config: GameConfig, players: List[Player]) -> Dict[str, Any]:
        """处理包赔逻辑"""
        compensation_config = game_config.compensation_config

        if compensation_config.scope == "不包赔":
            self.logger.info("无包赔规则")
            return {"triggered": False, "details": []}

        self.logger.info("\n--- 包赔规则检查 ---")

        compensation_details = []
        conditions = compensation_config.conditions

        for team in teams:
            for i, player_id in enumerate(team["team_players"]):
                net_score = team["net_scores"][i]
                teammate_net_score = team["net_scores"][1 - i]  # 队友净杆数

                # 检查包赔条件
                reasons = []

                if conditions.double_par and net_score >= hole.par * 2:
                    reasons.append("双par及以上")

                if conditions.plus_three and net_score >= hole.par + 3:
                    reasons.append("+3及以上")

                if conditions.diff_three and abs(net_score - teammate_net_score) >= 3:
                    reasons.append("与队友相差3杆及以上")

                if reasons:
                    # 找到队友ID
                    teammate_id = team["team_players"][1 - i]

                    compensation_details.append({
                        "player_id": player_id,
                        "reason": ", ".join(reasons),
                        "compensation_by": teammate_id
                    })

                    player_name = next((p.name for p in players if p.id == player_id), player_id)
                    teammate_name = next((p.name for p in players if p.id == teammate_id), teammate_id)

                    self.logger.info(f"  {player_name} 触发包赔: {', '.join(reasons)} (由{teammate_name}包赔)")

        return {
            "triggered": len(compensation_details) > 0,
            "details": compensation_details
        }

    def _handle_tie_logic(self, pk_results: Dict[str, Any], scoring_state: Dict[str, Any],
                          game_config: GameConfig, hole: Hole, teams: List[Dict[str, Any]]) -> Dict[str, Any]:
        """处理平洞逻辑"""
        tie_config = game_config.tie_config

        # 添加详细的调试日志
        self.logger.info(f"\n=== 平洞逻辑调试 ===")
        self.logger.info(
            f"当前平洞状态: 平洞次数={scoring_state['tie_count']}, 累积分数={scoring_state['tie_accumulated_score']}")
        self.logger.info(f"平洞配置: 计分规则={tie_config.scoring}, 收取规则={tie_config.collect_rule}")

        # 检查是否为平洞
        is_tie = self._check_tie_condition(pk_results, tie_config)
        self.logger.info(f"本洞是否为平洞: {is_tie}")

        if is_tie:
            # 获取平洞分数
            tie_score = self._get_tie_score(tie_config)
            scoring_state["tie_accumulated_score"] += tie_score
            scoring_state["tie_count"] += 1

            self.logger.info(f"\n--- 平洞处理 ---")
            self.logger.info(f"本洞平洞，累积{tie_score}分")
            self.logger.info(
                f"当前累积平洞分数: {scoring_state['tie_accumulated_score']}分 (共{scoring_state['tie_count']}次)")

            return {
                "is_tie": True,
                "accumulated_count": scoring_state["tie_count"],
                "accumulated_score": scoring_state["tie_accumulated_score"],
                "collected_score": 0,
                "collect_count": 0
            }
        else:
            # 不是平洞，检查是否需要收取累积分数
            collected_score = 0
            collect_count = 0

            # 修改这里：对于翻倍模式检查tie_count，对于非翻倍模式检查tie_accumulated_score
            scoring_rule = tie_config.scoring
            should_collect = False

            self.logger.info(f"非平洞，检查是否收取: 计分规则={scoring_rule}")

            if scoring_rule in ["平洞翻倍(不算鸟鹰奖励)", "平洞翻倍(算鸟鹰奖励)"]:
                should_collect = scoring_state["tie_count"] > 0  # 翻倍模式检查次数
                self.logger.info(f"翻倍模式检查: 平洞次数={scoring_state['tie_count']}, 是否收取={should_collect}")
            else:
                should_collect = scoring_state["tie_accumulated_score"] > 0  # 非翻倍模式检查分数
                self.logger.info(
                    f"非翻倍模式检查: 累积分数={scoring_state['tie_accumulated_score']}, 是否收取={should_collect}")

            if should_collect:
                self.logger.info("开始收取平洞分数...")
                collected_score, collect_count = self._collect_tie_score(pk_results, scoring_state, tie_config, hole,
                                                                         teams)
                self.logger.info(f"收取结果: 收取分数={collected_score}, 收取次数={collect_count}")

                if collected_score > 0:
                    self.logger.info(f"\n--- 平洞收取 ---")
                    self.logger.info(f"收取{collect_count}次平洞分数，共{collected_score}分")
                    self.logger.info(f"剩余累积平洞分数: {scoring_state['tie_accumulated_score']}分")
            else:
                self.logger.info("无需收取平洞分数")

            self.logger.info(f"=== 平洞逻辑调试结束 ===\n")

            return {
                "is_tie": False,
                "accumulated_count": scoring_state["tie_count"],
                "accumulated_score": scoring_state["tie_accumulated_score"],
                "collected_score": collected_score,
                "collect_count": collect_count
            }

    def _check_tie_condition(self, pk_results: Dict[str, Any], tie_config: Dict[str, Any]) -> bool:
        """检查平洞条件"""
        definition = tie_config.definition

        if definition == "得分差为0":
            # 三项比较结果总得分差为0
            total_diff = 0
            for pk_result in pk_results.values():
                if pk_result["winner_team"] == 1:
                    total_diff += pk_result["score_diff"]
                elif pk_result["winner_team"] == 2:
                    total_diff -= pk_result["score_diff"]

            return total_diff == 0

        return False

    def _get_tie_score(self, tie_config: Dict[str, Any]) -> int:
        """获取平洞分数"""
        scoring = tie_config.scoring

        score_map = {
            "平洞跳过(无肉)": 0,
            "平洞算1点": 1,
            "平洞算2点": 2,
            "平洞算3点": 3,
            "平洞算4点": 4,
            "平洞翻倍(不算鸟鹰奖)": 0,  # 翻倍模式累积次数，不累积固定分数
            "平洞翻倍(算鸟鹰奖)": 0,  # 翻倍模式累积次数，不累积固定分数
            "平洞连续翻番": 0  # 连续翻番模式累积次数
        }

        return score_map.get(scoring, 0)

    def _collect_tie_score(self, pk_results: Dict[str, Any], scoring_state: Dict[str, Any],
                           tie_config: Dict[str, Any], hole: Hole, teams: List[Dict[str, Any]]) -> Tuple[int, int]:
        """收取平洞分数，返回(收取分数, 收取次数)"""
        collect_rule = tie_config.collect_rule
        scoring_rule = tie_config.scoring

        self.logger.info(f"平洞分数收取调用: 收取规则={collect_rule}, 计分规则={scoring_rule}")

        # 处理不同的收取规则
        if collect_rule == "赢了全收掉":
            if scoring_rule in ["平洞翻倍(不算鸟鹰奖励)", "平洞翻倍(算鸟鹰奖励)"]:
                # 翻倍模式：收取分数 = 本洞PK净胜分 × 平洞次数
                if scoring_state["tie_count"] > 0:
                    # 计算本洞PK净胜分
                    total_diff = 0
                    for pk_result in pk_results.values():
                        if pk_result["winner_team"] == 1:
                            total_diff += pk_result["score_diff"]
                        elif pk_result["winner_team"] == 2:
                            total_diff -= pk_result["score_diff"]

                    base_score = abs(total_diff)  # 使用绝对值作为基础分数
                    collected = base_score * scoring_state["tie_count"]
                    collect_count = scoring_state["tie_count"]

                    self.logger.info(
                        f"翻倍模式收取: 本洞净胜分={total_diff}, 基础分数={base_score}, 平洞次数={scoring_state['tie_count']}")
                    self.logger.info(f"计算结果: 收取分数={collected}, 收取次数={collect_count}")

                    # 清空累积
                    scoring_state["tie_accumulated_score"] = 0
                    scoring_state["tie_count"] = 0

                    return collected, collect_count
            else:
                # 非翻倍模式：收取累积的固定分数
                collected = scoring_state["tie_accumulated_score"]
                collect_count = scoring_state["tie_count"]
                self.logger.info(f"非翻倍模式收取: 收取分数={collected}, 收取次数={collect_count}")
                scoring_state["tie_accumulated_score"] = 0
                scoring_state["tie_count"] = 0
                return collected, collect_count

        elif collect_rule == "Par收1/鸟收2/鹰收5":
            # 根据获胜队伍的最好成就确定收取倍数
            if scoring_rule in ["平洞翻倍(不算鸟鹰奖励)", "平洞翻倍(算鸟鹰奖励)"]:
                if scoring_state["tie_count"] > 0:
                    # 计算本洞PK净胜分
                    total_diff = 0
                    winner_team_id = 0
                    for pk_result in pk_results.values():
                        if pk_result["winner_team"] == 1:
                            total_diff += pk_result["score_diff"]
                        elif pk_result["winner_team"] == 2:
                            total_diff -= pk_result["score_diff"]

                    # 确定获胜队伍
                    if total_diff > 0:
                        winner_team_id = 1
                    elif total_diff < 0:
                        winner_team_id = 2

                    # 获取获胜队伍的最好成就
                    collect_multiplier = 1  # 默认按Par收1
                    if winner_team_id > 0:
                        winner_team = teams[winner_team_id - 1]
                        best_achievement = self._get_best_achievement_in_team(winner_team, hole.par)

                        # 根据最好成就确定收取倍数
                        if best_achievement == "小鸟":
                            collect_multiplier = 2
                        elif best_achievement == "老鹰":
                            collect_multiplier = 5
                        elif best_achievement == "信天翁":
                            collect_multiplier = 5  # HIO当作信天翁处理
                        else:
                            collect_multiplier = 1  # Par或更差

                        self.logger.info(f"获胜队伍最好成就: {best_achievement}, 期望收取倍数: {collect_multiplier}")

                    # 关键修复：收取次数不能超过实际积累的平洞次数
                    base_score = abs(total_diff)  # 使用绝对值作为基础分数
                    actual_collect_count = min(collect_multiplier, scoring_state["tie_count"])
                    collected = base_score * actual_collect_count

                    self.logger.info(
                        f"实际收取次数: {actual_collect_count} (期望{collect_multiplier}次，积累{scoring_state['tie_count']}次)")
                    self.logger.info(
                        f"计算: 本洞净胜分{base_score} × 实际收取次数{actual_collect_count} = {collected}分")

                    # 清空累积 - 按实际收取次数减少
                    scoring_state["tie_accumulated_score"] = 0
                    scoring_state["tie_count"] -= actual_collect_count

                    return collected, actual_collect_count

        self.logger.info("未满足收取条件，返回0")
        return 0, 0

    def _get_best_achievement_in_team(self, team: Dict[str, Any], par: int) -> str:
        """获取队伍中的最好成就"""
        best_score_to_par = float('inf')

        for i, player_id in enumerate(team["team_players"]):
            raw_score = team["raw_scores"][i]
            score_to_par = raw_score - par
            if score_to_par < best_score_to_par:
                best_score_to_par = score_to_par

        return self._get_achievement_name(best_score_to_par)

    def _get_best_achievement_in_team(self, team: Dict[str, Any], par: int) -> str:
        """获取队伍中的最好成就"""
        best_score_to_par = float('inf')

        for i, player_id in enumerate(team["team_players"]):
            raw_score = team["raw_scores"][i]
            score_to_par = raw_score - par
            if score_to_par < best_score_to_par:
                best_score_to_par = score_to_par

        return self._get_achievement_name(best_score_to_par)

    def _calculate_hole_scores(self, teams: List[Dict[str, Any]], pk_results: Dict[str, Any],
                               special_rewards: Dict[str, Any], compensation: Dict[str, Any],
                               tie_status: Dict[str, Any], game_config: GameConfig,
                               players: List[Player]) -> List[Dict[str, Any]]:
        """计算最终洞次得分"""
        self.logger.info("\n--- 最终洞次得分计算 ---")

        hole_scores = []

        # 初始化每个选手的得分
        for team in teams:
            for player_id in team["team_players"]:
                hole_scores.append({
                    "player_id": player_id,
                    "base_score": 0,
                    "bird_eagle_score": 0,
                    "double_kill_score": 0,
                    "tie_score": 0,
                    "compensation_score": 0,
                    "final_score": 0
                })

        # 1. 计算基础PK得分
        self._apply_base_pk_scores(hole_scores, teams, pk_results)

        # 2. 应用鸟鹰分数
        self._apply_bird_eagle_scores(hole_scores, special_rewards["bird_eagle"])

        # 3. 应用双杀奖励
        self._apply_double_kill_reward(hole_scores, teams, special_rewards["double_kill"], pk_results)

        # 4. 应用平洞分数
        self._apply_tie_scores(hole_scores, teams, tie_status, pk_results)

        # 5. 处理包赔调整
        self._apply_compensation_adjustment(hole_scores, teams, compensation)

        # 6. 计算最终分数 - 这里是新添加的关键部分
        for score in hole_scores:
            score["final_score"] = (score["base_score"] +
                                    score["bird_eagle_score"] +
                                    score["double_kill_score"] +
                                    score["tie_score"] +
                                    score["compensation_score"])

        # 7. 打印每个选手的得分详情
        for score in hole_scores:
            # 查找选手姓名
            player_name = next((p.name for p in players if p.id == score["player_id"]), score["player_id"])

            components = []
            if score["bird_eagle_score"] != 0:
                # 有鸟鹰奖励时，显示为鸟鹰分数
                components.append(f"鸟鹰{score['base_score']}")
            else:
                # 无鸟鹰奖励时，显示为基础分数
                if score["base_score"] != 0:
                    components.append(f"基础{score['base_score']}")

            if score["double_kill_score"] != 0:
                components.append(f"双杀{score['double_kill_score']}")
            if score["tie_score"] != 0:
                components.append(f"平洞{score['tie_score']}")
            if score["compensation_score"] != 0:
                components.append(f"包赔{score['compensation_score']}")

            components_str = " + ".join(components) if components else "0"
            self.logger.info(f"  {player_name}: {components_str} = {score['final_score']}分")

        return hole_scores

    def _apply_base_pk_scores(self, hole_scores: List[Dict[str, Any]], teams: List[Dict[str, Any]],
                              pk_results: Dict[str, Any]):
        """应用基础PK得分"""
        # 计算每队在三项比较中的总得分
        team_base_scores = {1: 0, 2: 0}

        for pk_name, pk_result in pk_results.items():
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
        """应用鸟鹰分数（替换基础分数）"""
        # 找出有鸟鹰奖励的队伍
        team_bird_eagle_scores = {}

        for reward in bird_eagle_rewards:
            if reward["applied"] and reward["final_bird_eagle_score"] > 0:
                player_id = reward["player_id"]
                bird_eagle_score = reward["final_bird_eagle_score"]

                # 找到该选手所在的队伍，记录队伍的鸟鹰分数
                for score in hole_scores:
                    if score["player_id"] == player_id:
                        team_bird_eagle_scores[player_id] = bird_eagle_score
                        break

        # 如果有鸟鹰奖励，则替换所有选手的基础分数
        if team_bird_eagle_scores:
            # 获取鸟鹰队伍的分数
            bird_eagle_score = list(team_bird_eagle_scores.values())[0]

            for score in hole_scores:
                if score["player_id"] in team_bird_eagle_scores:
                    # 鸟鹰队伍：基础分被鸟鹰分替换
                    score["base_score"] = bird_eagle_score  # 直接替换为鸟鹰分数
                    score["bird_eagle_score"] = 0  # 不需要额外增量
                else:
                    # 对手队伍：基础分变成负的鸟鹰分
                    score["base_score"] = -bird_eagle_score

    def _apply_double_kill_reward(self, hole_scores: List[Dict[str, Any]], teams: List[Dict[str, Any]],
                                  double_kill: Dict[str, Any], pk_results: Dict[str, Any]):
        """应用双杀奖励"""
        if not double_kill.get("triggered", False):
            return

        winner_team_id = double_kill["team_id"]
        reward_points = double_kill["reward_points"]

        if reward_points == -1:  # 翻倍奖励
            # 获胜队伍本洞所有得分翻倍
            for team in teams:
                if team["team_id"] == winner_team_id:
                    for player_id in team["team_players"]:
                        for score in hole_scores:
                            if score["player_id"] == player_id:
                                # 翻倍基础分和鸟鹰分
                                original_total = score["base_score"] + score["bird_eagle_score"]
                                score["double_kill_score"] = original_total  # 翻倍效果
                                break
        else:
            # 固定分数奖励
            for team in teams:
                multiplier = reward_points if team["team_id"] == winner_team_id else -reward_points
                for player_id in team["team_players"]:
                    for score in hole_scores:
                        if score["player_id"] == player_id:
                            score["double_kill_score"] = multiplier
                            break

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

    def _apply_compensation_adjustment(self, hole_scores: List[Dict[str, Any]], teams: List[Dict[str, Any]],
                                       compensation: Dict[str, Any]):
        """应用包赔调整"""
        if not compensation.get("triggered", False):
            return

        self.logger.info("\n--- 包赔调整 ---")

        for detail in compensation["details"]:
            compensating_player = detail["player_id"]
            compensated_by = detail["compensation_by"]

            # 找到对应的得分记录
            compensating_score = None
            compensated_by_score = None

            for score in hole_scores:
                if score["player_id"] == compensating_player:
                    compensating_score = score
                elif score["player_id"] == compensated_by:
                    compensated_by_score = score

            if compensating_score and compensated_by_score:
                # 计算被包赔选手的负分总和
                compensating_negative = min(0, compensating_score["base_score"] +
                                            compensating_score["bird_eagle_score"] +
                                            compensating_score["double_kill_score"] +
                                            compensating_score["tie_score"])

                if compensating_negative < 0:
                    # 包赔选手承担负分
                    compensated_by_score["compensation_score"] += compensating_negative
                    compensating_score["compensation_score"] -= compensating_negative

                    self.logger.info(
                        f"  {compensating_player} 的 {-compensating_negative}分负分由 {compensated_by} 承担")

    def _determine_next_tee_order(self, hole: Hole, teams: List[Dict[str, Any]], players: List[Player]) -> List[str]:
        """确定下洞开球顺序"""
        self.logger.info(f"\n--- 确定下洞开球顺序 ---")

        # 收集所有选手的净杆数和当前洞开球顺序位置
        player_scores = []
        current_tee_order = hole.tee_order

        for team in teams:
            for i, player_id in enumerate(team["team_players"]):
                net_score = team["net_scores"][i]
                # 获取该选手在当前洞开球顺序中的位置
                current_position = current_tee_order.index(player_id)
                player_scores.append((player_id, net_score, current_position))

        # 打印当前洞的净杆数详情
        self.logger.info("本洞净杆数详情:")
        for player_id, net_score, position in player_scores:
            player_name = next((p.name for p in players if p.id == player_id), player_id)
            self.logger.info(f"  {player_name}: {net_score}杆 (本洞开球第{position + 1}位)")

        # 按净杆数升序排序，净杆数相同时按当前洞开球顺序排序
        player_scores.sort(key=lambda x: (x[1], x[2]))

        # 打印排序逻辑详情
        self.logger.info("下洞开球顺序排序逻辑:")
        for i, (player_id, net_score, position) in enumerate(player_scores):
            player_name = next((p.name for p in players if p.id == player_id), player_id)
            self.logger.info(f"  第{i + 1}位: {player_name} ({net_score}杆)")

        next_order = [player_id for player_id, _, _ in player_scores]

        # 打印最终开球顺序
        next_order_names = [next((p.name for p in players if p.id == pid), pid) for pid in next_order]
        self.logger.info(f"下洞开球顺序: {', '.join(next_order_names)}")

        return next_order

    def _get_players_from_teams(self, teams: List[Dict[str, Any]]) -> List:
        """从队伍数据中获取选手信息的辅助方法"""
        # 这需要访问game_data.players，需要修改函数签名传入players参数
        # 或者在类中保存players引用
        pass

    def _update_scoring_state(self, scoring_state: Dict[str, Any], hole_result: Dict[str, Any]):
        """更新累积计分状态"""
        # 更新选手总分
        for score in hole_result["hole_scores"]:
            player_id = score["player_id"]
            scoring_state["player_total_scores"][player_id] += score["final_score"]

    def _calculate_final_summary(self, game_data: LaSiGameData, scoring_state: Dict[str, Any],
                                 hole_details: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算最终总结"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("最终结算")
        self.logger.info("=" * 60)

        # 基础总分
        final_scores = []
        for player in game_data.players:
            total_score = scoring_state["player_total_scores"][player.id]
            final_scores.append({
                "player_id": player.id,
                "player_name": player.name,
                "total_score": total_score,
                "after_donation": total_score,
                "after_adjustment": total_score
            })

        # 打印基础总分
        self.logger.info("\n基础总分:")
        for score in final_scores:
            self.logger.info(f"  {score['player_name']}: {score['total_score']}分")

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

        # 计算让分调整
        adjustment_summary = self._calculate_score_adjustment(final_scores, game_data.game_config)

        # 打印最终得分
        self.logger.info("\n最终得分:")
        for score in final_scores:
            self.logger.info(f"  {score['player_name']}: {score['after_adjustment']}分")

        return {
            "total_holes": len(game_data.holes),
            "game_mode": game_data.game_config.mode,
            "final_scores": final_scores,
            "donation_summary": donation_summary,
            "adjustment_summary": adjustment_summary
        }

    def _calculate_donation(self, final_scores: List[Dict[str, Any]], game_config: GameConfig) -> Dict[str, Any]:
        """计算捐锅"""
        donation_config = game_config.donation_config.type
        donation_details = []
        total_donated = 0

        if donation_config == "不捐":
            self.logger.info("\n无捐锅规则")
        else:
            self.logger.info(f"\n捐锅规则: {donation_config}")

            if donation_config == "赢了捐1点":
                for score in final_scores:
                    if score["total_score"] > 0:
                        donated = 1
                        donation_details.append({"player_id": score["player_id"], "donated": donated})
                        total_donated += donated
                        self.logger.info(f"  {score['player_name']} 捐锅: {donated}分")
            elif donation_config == "每赢2点捐1点":
                for score in final_scores:
                    if score["total_score"] > 0:
                        donated = score["total_score"] // 2
                        if donated > 0:
                            donation_details.append({"player_id": score["player_id"], "donated": donated})
                            total_donated += donated
                            self.logger.info(f"  {score['player_name']} 捐锅: {donated}分")
            elif donation_config == "每赢3点捐1点":
                for score in final_scores:
                    if score["total_score"] > 0:
                        donated = score["total_score"] // 3
                        if donated > 0:
                            donation_details.append({"player_id": score["player_id"], "donated": donated})
                            total_donated += donated
                            self.logger.info(f"  {score['player_name']} 捐锅: {donated}分")
            elif donation_config == "赢了全捐":
                for score in final_scores:
                    if score["total_score"] > 0:
                        donated = score["total_score"]
                        donation_details.append({"player_id": score["player_id"], "donated": donated})
                        total_donated += donated
                        self.logger.info(f"  {score['player_name']} 捐锅: {donated}分")

        return {
            "total_donated": total_donated,
            "donation_details": donation_details
        }

    def _calculate_score_adjustment(self, final_scores: List[Dict[str, Any]], game_config: GameConfig) -> Dict[
        str, Any]:
        """计算让分调整"""
        adjustment_config = game_config.score_adjustment_config
        mode = adjustment_config.mode
        adjustment_type = adjustment_config.adjustment_type
        points = adjustment_config.points

        if points == 0 or game_config.mode != "固拉":
            return {
                "mode": mode,
                "type": adjustment_type,
                "points": points,
                "applied": False
            }

        self.logger.info(f"\n让分调整: {mode} {adjustment_type} {points}分")

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
                            self.logger.info(f"  {score['player_name']} 获得让分: +{points}分")
                            break
            elif adjustment_type == "虚让":
                # 虚让：如果差距小于等于让分，算平局
                if score_diff <= points:
                    # 所有人分数设为0（平局）
                    for score in final_scores:
                        score["after_adjustment"] = 0
                    applied = True
                    self.logger.info("  差距小于让分，算平局")
        elif mode == "互虚":
            # 互虚：双向虚让
            if score_diff <= points:
                for score in final_scores:
                    score["after_adjustment"] = 0
                applied = True
                self.logger.info("  差距小于互虚分数，算平局")

        return {
            "mode": mode,
            "type": adjustment_type,
            "points": points,
            "applied": applied
        }

    def _create_error_response(self, message: str) -> Dict[str, Any]:
        """创建错误响应"""
        return {
            "success": False,
            "error": {
                "code": "CALCULATION_ERROR",
                "message": message
            }
        }


def calculate_lasi_score(game_data: LaSiGameData) -> Dict[str, Any]:
    """拉丝三点计分接口函数"""
    scoring_system = LaSiThreePointScoring()
    return scoring_system.calculate_score(game_data)


# 测试示例
if __name__ == "__main__":
    # 这里可以添加测试代码
    pass

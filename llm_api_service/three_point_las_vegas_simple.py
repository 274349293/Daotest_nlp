import json
import copy
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from utils.nlp_logging import CustomLogger

"""
拉丝三点高尔夫击球顺序计算接口
专门用于计算乱拉模式下每洞的击球顺序（tee_order）

功能：
- 基于上一洞净杆数计算下一洞击球顺序
- 应用让杆规则和让杆限制条件
- 只处理乱拉模式的默认配对规则
- 填充并返回完整的tee_order数据
"""

logger = CustomLogger(name="DaoTest Tee Order Calculator", write_to_file=True)


# 使用与lasi_scoring相同的数据模型
class FixedTeam(BaseModel):
    team_id: int
    players: List[str]


class CombinationPKConfig(BaseModel):
    mode: str = Field(default="双方总杆相加PK", description="双方总杆相加PK | 双方总杆相乘PK")


class ExpertSeparationConfig(BaseModel):
    enabled: bool = Field(default=False, description="是否启用高手不见面")
    expert_players: List[str] = Field(default_factory=list, description="高手玩家ID列表，必须2人")


class DoubleKillConfig(BaseModel):
    type: str = Field(default="不奖励", description="不奖励|奖励1分|奖励2分|奖励3分|翻倍奖励")


class BirdEagleConfig(BaseModel):
    condition: str = Field(default="合并pk赢了才奖励", description="鸟鹰奖励条件")
    extra_reward: str = Field(default="不奖励", description="额外奖励类型")


class TieConfig(BaseModel):
    definition: str = Field(default="得分差为0", description="平洞定义")
    scoring: str = Field(default="平洞跳过(无肉)", description="平洞计分规则")
    collect_rule: str = Field(default="赢了全收掉", description="平洞收取规则")


class CompensationConditions(BaseModel):
    double_par: bool = Field(default=False, description="双par及以上")
    plus_three: bool = Field(default=False, description="+3及以上")
    diff_three: bool = Field(default=False, description="相差3杆及以上")


class CompensationConfig(BaseModel):
    scope: str = Field(default="不包赔", description="不包赔|包本洞所有分|包含平洞的所有分")
    conditions: CompensationConditions = Field(default_factory=CompensationConditions)


class DonationConfig(BaseModel):
    type: str = Field(default="不捐", description="不捐|赢了捐1点|每赢2点捐1点|每赢3点捐1点|赢了全捐")


class HandicapRestrictions(BaseModel):
    only_combination_pk: bool = Field(default=False, description="仅组合PK让杆")
    no_leader: bool = Field(default=False, description="打头不让杆")
    no_par_bird_eagle: bool = Field(default=False, description="Par/鸟/鹰不让杆")


class HandicapConfig(BaseModel):
    restrictions: HandicapRestrictions = Field(default_factory=HandicapRestrictions)


class ScoreAdjustmentConfig(BaseModel):
    mode: str = Field(default="单让", description="单让|互虚")
    adjustment_type: str = Field(default="实让", description="实让|虚让")
    points: int = Field(default=0, description="让分数")


class GameConfig(BaseModel):
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
    id: str
    name: str


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
    holes: List[Hole]


class TeeOrderCalculator:
    """击球顺序计算器"""

    def __init__(self):
        self.logger = logger

    def calculate_tee_order(self, game_data: LaSiGameData) -> LaSiGameData:
        """计算击球顺序主函数"""
        try:
            self.logger.info("=== 开始计算击球顺序 ===")

            # 数据验证
            self._validate_game_data(game_data)

            # 深拷贝避免修改原数据
            result_data = copy.deepcopy(game_data)

            # 逐洞计算击球顺序
            for i, hole in enumerate(result_data.holes):
                if i == 0:
                    # 第一洞：验证tee_order是否完整
                    if len(hole.tee_order) != 4:
                        raise ValueError(f"第1洞的tee_order必须包含4个选手，当前为: {len(hole.tee_order)}")
                    self.logger.info(
                        f"第1洞击球顺序: {[self._get_player_name(pid, result_data.players) for pid in hole.tee_order]}")
                else:
                    # 后续洞次：基于上一洞净杆数计算
                    previous_hole = result_data.holes[i - 1]
                    next_tee_order = self._calculate_next_tee_order(
                        previous_hole, result_data.game_config, result_data.players
                    )
                    hole.tee_order = next_tee_order

                    player_names = [self._get_player_name(pid, result_data.players) for pid in next_tee_order]
                    self.logger.info(f"第{hole.hole_number}洞击球顺序: {player_names}")

            self.logger.info("=== 击球顺序计算完成 ===")
            return result_data

        except Exception as e:
            self.logger.error(f"计算击球顺序时发生错误: {str(e)}")
            raise

    def _validate_game_data(self, game_data: LaSiGameData):
        """验证游戏数据"""
        # 验证游戏模式
        if game_data.game_config.mode != "乱拉":
            raise ValueError(f"当前只支持乱拉模式，输入的模式为: {game_data.game_config.mode}")

        # 验证玩家数量
        if len(game_data.players) != 4:
            raise ValueError(f"必须有4名选手，当前为: {len(game_data.players)}名")

        # 验证洞次数据
        if not game_data.holes:
            raise ValueError("洞次数据不能为空")

        # 验证每洞分数数据
        for hole in game_data.holes:
            if len(hole.scores) != 4:
                raise ValueError(f"第{hole.hole_number}洞分数数据不完整，需要4名选手的分数")

        # 验证第一洞的tee_order
        first_hole = game_data.holes[0]
        if len(first_hole.tee_order) != 4:
            raise ValueError("第1洞必须提供完整的击球顺序（4名选手）")

        # 验证选手ID一致性
        player_ids = {p.id for p in game_data.players}
        tee_order_ids = set(first_hole.tee_order)
        if tee_order_ids != player_ids:
            raise ValueError("第1洞击球顺序中的选手ID与选手列表不匹配")

        self.logger.info("数据验证通过")

    def _calculate_next_tee_order(self, previous_hole: Hole, game_config: GameConfig,
                                  players: List[Player]) -> List[str]:
        """计算下一洞击球顺序"""
        self.logger.info(f"基于第{previous_hole.hole_number}洞计算下一洞击球顺序")

        # 构建分数映射
        score_map = {score.player_id: score.raw_strokes for score in previous_hole.scores}

        # 计算每个选手的净杆数
        player_net_scores = []
        for player in players:
            raw_score = score_map[player.id]

            # 获取让杆数
            handicap = self._get_player_handicap(
                player.id, previous_hole.hole_type, game_config.handicap_settings
            )

            # 应用让杆限制条件
            effective_handicap = self._apply_handicap_restrictions(
                player.id, handicap, raw_score, previous_hole.par,
                score_map, game_config.handicap_config.restrictions
            )

            # 计算净杆数
            net_score = raw_score - effective_handicap

            # 获取上一洞的击球顺序位置（用于同分处理）
            previous_position = previous_hole.tee_order.index(player.id)

            player_net_scores.append({
                'player_id': player.id,
                'player_name': player.name,
                'raw_score': raw_score,
                'handicap': effective_handicap,
                'net_score': net_score,
                'previous_position': previous_position
            })

            if effective_handicap > 0:
                self.logger.info(f"  {player.name}: {raw_score}杆 - {effective_handicap}让杆 = {net_score}净杆")
            else:
                self.logger.info(f"  {player.name}: {raw_score}杆 (无让杆)")

        # 排序：按净杆数升序，同分时按上一洞击球顺序排序
        player_net_scores.sort(key=lambda x: (x['net_score'], x['previous_position']))

        # 生成下一洞击球顺序
        next_tee_order = [player_info['player_id'] for player_info in player_net_scores]

        # 打印排序详情
        self.logger.info("排序详情（净杆数，上洞位置）:")
        for i, player_info in enumerate(player_net_scores):
            self.logger.info(f"  第{i + 1}位: {player_info['player_name']} "
                             f"(净杆{player_info['net_score']}, 上洞第{player_info['previous_position'] + 1}位)")

        return next_tee_order

    def _get_player_handicap(self, player_id: str, hole_type: str,
                             handicap_settings: Dict[str, Dict[str, float]]) -> float:
        """获取选手让杆数"""
        if player_id in handicap_settings:
            return handicap_settings[player_id].get(hole_type, 0.0)
        return 0.0

    def _apply_handicap_restrictions(self, player_id: str, handicap: float, raw_score: int,
                                     par: int, all_scores: Dict[str, int],
                                     restrictions: HandicapRestrictions) -> float:
        """应用让杆限制条件"""
        player_name = player_id  # 简化显示，实际项目中可以查找真实姓名

        # 检查是否为该洞成绩最好的选手（打头不让）
        if restrictions.no_leader:
            min_raw_score = min(all_scores.values())
            if raw_score == min_raw_score:
                if handicap > 0:
                    self.logger.info(f"    {player_name} 打头不让杆 (原让杆: {handicap})")
                return 0.0

        # 检查是否打出Par/鸟/鹰（不让杆）
        if restrictions.no_par_bird_eagle:
            score_to_par = raw_score - par
            if score_to_par <= 0:  # Par或更好
                achievement = "标准杆" if score_to_par == 0 else ("小鸟" if score_to_par == -1 else "老鹰")
                if handicap > 0:
                    self.logger.info(f"    {player_name} 打出{achievement}不让杆 (原让杆: {handicap})")
                return 0.0

        return handicap

    def _get_player_name(self, player_id: str, players: List[Player]) -> str:
        """获取选手姓名"""
        for player in players:
            if player.id == player_id:
                return player.name
        return player_id


# 对外接口函数
def calculate_tee_order(game_data: LaSiGameData) -> LaSiGameData:
    """
    计算拉丝三点高尔夫击球顺序接口函数

    Args:
        game_data: 拉丝游戏数据，其中第一洞的tee_order需要完整，其他洞的tee_order可以为空

    Returns:
        LaSiGameData: 填充好所有洞tee_order的完整数据

    Raises:
        ValueError: 数据验证失败
        Exception: 其他计算错误
    """
    logger.info("------------------击球顺序计算开始--------------------")

    try:
        calculator = TeeOrderCalculator()
        result = calculator.calculate_tee_order(game_data)

        logger.info("击球顺序计算成功完成")
        return result

    except Exception as e:
        logger.error(f"击球顺序计算失败: {str(e)}")
        raise

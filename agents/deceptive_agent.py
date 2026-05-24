import numpy as np
from agents.base_agent import BaseAgent

ACTION_FOLD = 0
ACTION_CHECK_CALL = 1
ACTION_RAISE_MIN = 2
ACTION_RAISE_50 = 3
ACTION_RAISE_75 = 4
ACTION_RAISE_POT = 5
ACTION_RAISE_OVERBET = 6
ACTION_ALL_IN = 7

# obs indices
_POT_IDX = 104
_OPP_AF_IDX = 126   # opponent aggression factor (obs[122+2] = opp_stats AF)


class DeceptiveAgent(BaseAgent):
    """
    誘導型 Agent。
    策略特性：最大化對手決策錯誤，bluff / slowplay 混合策略。
    修正：hand_strength proxy 改用 pot_ratio + opp_af 組合，
          不再誤用 obs[0]（手牌 one-hot）。
    """

    def __init__(self, obs_size: int, action_size: int, bluff_frequency: float = 0.35, **kwargs):
        super().__init__(obs_size, action_size, **kwargs)
        self.bluff_frequency = bluff_frequency

    def compute_reward_shaping(
        self, action: int, obs: np.ndarray, base_reward: float
    ) -> float:
        pot_ratio = float(np.clip(obs[_POT_IDX], 0, 1))
        # opp_af: obs[126] = opponent AF (aggression factor, 0~1 normalized)
        opp_af = float(np.clip(obs[_OPP_AF_IDX] if len(obs) > _OPP_AF_IDX else 0.5, 0, 1))

        shaping = base_reward

        # 對手很被動（低 AF）→ bluff raise 更容易成功 → 加成
        if action in (ACTION_RAISE_50, ACTION_RAISE_75, ACTION_RAISE_POT, ACTION_RAISE_OVERBET):
            if opp_af < 0.4 and np.random.random() < self.bluff_frequency:
                shaping += 0.15 * pot_ratio

        # 對手很激進（高 AF）→ slowplay check/call 引誘 → 加成
        if action == ACTION_CHECK_CALL and opp_af > 0.6:
            shaping += 0.1 * pot_ratio

        return shaping

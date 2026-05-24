import numpy as np
from agents.base_agent import BaseAgent

ACTION_FOLD = 0
ACTION_CHECK_CALL = 1
ACTION_RAISE_MIN = 2
ACTION_RAISE_POT = 5
ACTION_RAISE_OVERBET = 6
ACTION_ALL_IN = 7


class AggressiveAgent(BaseAgent):
    """
    激進型 Agent。

    策略特性：
    - 高頻率 bet/raise，施加最大 fold equity
    - Reward shaping 獎勵 raise 行為，懲罰 passive check/call
    - 傾向 overbet 和 all-in 以製造壓力
    """

    def __init__(self, obs_size: int, action_size: int, **kwargs):
        super().__init__(obs_size, action_size, **kwargs)
        self.aggression_bonus = 0.15
        self.passive_penalty = -0.05

    def compute_reward_shaping(self, action: int, obs: np.ndarray, base_reward: float) -> float:
        shaping = base_reward
        if action in (ACTION_RAISE_MIN, ACTION_RAISE_POT, ACTION_RAISE_OVERBET, ACTION_ALL_IN):
            shaping += self.aggression_bonus
        elif action == ACTION_CHECK_CALL:
            shaping += self.passive_penalty
        elif action == ACTION_FOLD:
            shaping -= 0.1
        return shaping

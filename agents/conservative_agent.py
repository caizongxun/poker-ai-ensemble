import numpy as np
from agents.base_agent import BaseAgent

ACTION_FOLD = 0
ACTION_CHECK_CALL = 1
ACTION_RAISE_MIN = 2
ACTION_RAISE_OVERBET = 6
ACTION_ALL_IN = 7


class ConservativeAgent(BaseAgent):
    """
    保守型 Agent。

    策略特性：
    - 低 VPIP，只有強牌才進
    - Reward shaping 懲罰 overbet 和 all-in，獎勵小額 raise 和 check
    - 最小化 variance，傾向擇牌而非冒險
    """

    def __init__(self, obs_size: int, action_size: int, **kwargs):
        super().__init__(obs_size, action_size, **kwargs)
        self.overbet_penalty = -0.2
        self.safe_play_bonus = 0.05

    def compute_reward_shaping(self, action: int, obs: np.ndarray, base_reward: float) -> float:
        shaping = base_reward
        if action in (ACTION_RAISE_OVERBET, ACTION_ALL_IN):
            shaping += self.overbet_penalty
        elif action in (ACTION_CHECK_CALL, ACTION_RAISE_MIN):
            shaping += self.safe_play_bonus
        return shaping

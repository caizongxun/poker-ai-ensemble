import numpy as np
from agents.base_agent import BaseAgent

ACTION_FOLD = 0
ACTION_CHECK_CALL = 1
ACTION_RAISE_MIN = 2
ACTION_RAISE_OVERBET = 6
ACTION_ALL_IN = 7

# obs index for SPR (Stack to Pot Ratio), defined in poker_env.py obs[112]
_SPR_IDX = 112


class ConservativeAgent(BaseAgent):
    """
    保守型 Agent。
    策略特性：低 VPIP，只有強牌才進；最小化 variance。
    修正：移除 safe_play_bonus（避免學到一直 call 的 degenerate 策略）。
    Reward shaping 只懲罰高風險行為，不再獎勵 call。
    """

    def __init__(self, obs_size: int, action_size: int, **kwargs):
        super().__init__(obs_size, action_size, **kwargs)

    def compute_reward_shaping(
        self, action: int, obs: np.ndarray, base_reward: float
    ) -> float:
        # SPR 低代表籌碼已深陷，此時 overbet/all-in 風險極高
        spr = float(np.clip(obs[_SPR_IDX], 0, 1)) * 10  # 還原 0~10 範圍
        shaping = base_reward

        if action in (ACTION_RAISE_OVERBET, ACTION_ALL_IN):
            # SPR 高時還 all-in → 高風險懲罰
            penalty = 0.15 * (spr / 10.0)
            shaping -= penalty

        # 不再獎勵 call，避免 degenerate 策略
        return shaping

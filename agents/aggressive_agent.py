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
    策略特性：高頻率 bet/raise，施加最大 fold equity。
    Reward shaping 用 pot 比例，避免被 chip 數量淹沒。
    """

    def __init__(self, obs_size: int, action_size: int, **kwargs):
        super().__init__(obs_size, action_size, **kwargs)

    def compute_reward_shaping(
        self, action: int, obs: np.ndarray, base_reward: float
    ) -> float:
        # pot_ratio: obs[104] = pot / (stack_size * 2)，還原 pot 估算
        pot_ratio = float(np.clip(obs[104], 0, 1))
        pot_est = pot_ratio * 2  # 相對單位，0~2

        shaping = base_reward
        if action in (ACTION_RAISE_MIN, ACTION_RAISE_POT, ACTION_RAISE_OVERBET, ACTION_ALL_IN):
            # 依 pot 大小給予加成：pot 越大，aggressive raise 越有價值
            shaping += 0.1 * pot_est
        elif action == ACTION_CHECK_CALL:
            # passive call 輕微懲罰
            shaping -= 0.05 * pot_est
        elif action == ACTION_FOLD:
            # fold 懲罰（鼓勵施壓而非逃跑）
            shaping -= 0.1 * pot_est
        return shaping

import numpy as np
from agents.base_agent import BaseAgent

ACTION_FOLD = 0
ACTION_CHECK_CALL = 1
ACTION_RAISE_MIN = 2
ACTION_RAISE_POT = 5
ACTION_RAISE_OVERBET = 6
ACTION_ALL_IN = 7

# obs indices
_POT_IDX = 104   # pot / (stack_size * 2)
_OPP_AF_IDX = 126  # opponent aggression factor (normalized 0~1)


class AggressiveAgent(BaseAgent):
    """
    激進型 Agent。
    修正：
    - raise 獎勵改用 pot 比例，避免被 chip 量淹沒
    - 新增：raise 但輸授額懲罰（懲罰盲目 bluff）
    - 對手小牌也會 call的情況下，鼓勵收改起手大小
    """

    def __init__(self, obs_size: int, action_size: int, **kwargs):
        super().__init__(obs_size, action_size, **kwargs)

    def compute_reward_shaping(
        self, action: int, obs: np.ndarray, base_reward: float
    ) -> float:
        pot_ratio = float(np.clip(obs[_POT_IDX], 0, 1))
        pot_est = pot_ratio * 2  # 相對單位 0~2
        opp_af = float(np.clip(
            obs[_OPP_AF_IDX] if len(obs) > _OPP_AF_IDX else 0.5, 0, 1))

        shaping = base_reward

        if action in (ACTION_RAISE_MIN, ACTION_RAISE_POT,
                      ACTION_RAISE_OVERBET, ACTION_ALL_IN):
            if base_reward > 0:
                # raise 且贏：獎勵
                shaping += 0.1 * pot_est
            else:
                # raise 但輸：懲罰（懲罰盲目 bluff）
                # opp_af 高 = 對手也很激進 → 懲罰更大（不該打對攻）
                shaping -= 0.15 * pot_est * (0.5 + opp_af * 0.5)

        elif action == ACTION_CHECK_CALL:
            # passive call 輕微懲罰
            shaping -= 0.03 * pot_est

        elif action == ACTION_FOLD:
            shaping -= 0.08 * pot_est

        return shaping

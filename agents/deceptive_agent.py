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


class DeceptiveAgent(BaseAgent):
    """
    誘導型 Agent。

    策略特性：
    - 最大化對手的決策錯誤
    - 弱牌 bluff raise，強牌 slowplay（反向行為）
    - Mixed strategy：同樣牌面不固定打法，增加不可預測性
    - Reward shaping：若對手做出 -EV 決策則給予 bonus
    """

    def __init__(self, obs_size: int, action_size: int, bluff_frequency: float = 0.35, **kwargs):
        super().__init__(obs_size, action_size, **kwargs)
        self.bluff_frequency = bluff_frequency
        self.deception_bonus = 0.25

    def compute_reward_shaping(self, action: int, obs: np.ndarray, base_reward: float) -> float:
        shaping = base_reward
        hand_strength = float(np.clip(obs[0], 0, 1))

        # 成功的 bluff：手牌弱但對手 fold
        if action in (ACTION_RAISE_50, ACTION_RAISE_75, ACTION_RAISE_POT, ACTION_RAISE_OVERBET):
            if hand_strength < 0.4 and np.random.random() < self.bluff_frequency:
                shaping += self.deception_bonus

        # slowplay：手牌很強但選擇 check/call
        if action == ACTION_CHECK_CALL and hand_strength > 0.8:
            shaping += self.deception_bonus * 0.5

        return shaping

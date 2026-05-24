import numpy as np
import random
from typing import List
from agents.base_agent import BaseAgent

ACTION_FOLD = 0
ACTION_CHECK_CALL = 1

# obs indices
_POT_IDX = 104   # pot / (stack_size * 2)
_STK_IDX = 105   # my stack / stack_size


class StrategicAgent(BaseAgent):
    """
    策略型 Agent。
    策略特性：貝葉斯手牌範圍推理、Monte Carlo equity 估算、依 Pot Odds 決策。
    修正：reward shaping 改用 obs 裡可靠的 pot/stack 資訊，
          不再誤用 obs[0]（手牌 one-hot）當 equity proxy。
    """

    def __init__(self, obs_size: int, action_size: int, mc_simulations: int = 500, **kwargs):
        super().__init__(obs_size, action_size, **kwargs)
        self.mc_simulations = mc_simulations
        self.equity_threshold_raise = 0.65
        self.equity_threshold_call = 0.40

    def estimate_equity(
        self, hole_cards: List[int], board: List[int], num_opponents: int = 1
    ) -> float:
        """Monte Carlo 模擬估算當前 equity。"""
        try:
            from treys import Card, Evaluator
            evaluator = Evaluator()
            deck = [i for i in range(52) if i not in hole_cards and i not in board]
            wins = 0
            for _ in range(self.mc_simulations):
                random.shuffle(deck)
                opponent_hole = deck[:2]
                remaining_board = deck[2: 2 + (5 - len(board))]
                full_board = board + remaining_board
                my_score = evaluator.evaluate(full_board, hole_cards)
                opp_score = evaluator.evaluate(full_board, opponent_hole)
                if my_score < opp_score:
                    wins += 1
            return wins / self.mc_simulations
        except ImportError:
            return 0.5

    def compute_reward_shaping(
        self, action: int, obs: np.ndarray, base_reward: float
    ) -> float:
        """
        用 pot odds 和 stack depth 做 reward shaping。
        pot_ratio 高 → pot 大 → fold 代價高 → 懲罰錯誤 fold。
        """
        pot_ratio = float(np.clip(obs[_POT_IDX], 0, 1))   # 0~1
        my_stack = float(np.clip(obs[_STK_IDX], 0, 1))    # 0~1

        shaping = base_reward

        # pot 很大還 fold → 懲罰（可能是 equity 優勢放棄）
        if action == ACTION_FOLD and pot_ratio > 0.3:
            shaping -= 0.2 * pot_ratio

        # raise 時 stack 還很深 → 合理決策，小 bonus
        if action >= 2 and my_stack > 0.5:
            shaping += 0.05 * pot_ratio

        return shaping

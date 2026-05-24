import numpy as np
import random
from typing import List
from agents.base_agent import BaseAgent

ACTION_FOLD = 0
ACTION_CHECK_CALL = 1


class StrategicAgent(BaseAgent):
    """
    策略型 Agent。

    策略特性：
    - 貝葉斯手牌範圍推理，縮小對手 range
    - Monte Carlo 模擬計算當前 equity
    - 依 Pot Odds 決策，追求 EV+ 的每一步
    - Reward shaping 依 equity 優勢動態調整
    """

    def __init__(self, obs_size: int, action_size: int, mc_simulations: int = 500, **kwargs):
        super().__init__(obs_size, action_size, **kwargs)
        self.mc_simulations = mc_simulations
        self.equity_threshold_raise = 0.65
        self.equity_threshold_call = 0.40

    def estimate_equity(self, hole_cards: List[int], board: List[int], num_opponents: int = 1) -> float:
        """
        Monte Carlo 模擬估算當前 equity。
        hole_cards: list of card indices (0-51)
        board: list of community card indices
        """
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
                if my_score < opp_score:  # treys: lower is better
                    wins += 1
            return wins / self.mc_simulations
        except ImportError:
            return 0.5

    def compute_reward_shaping(self, action: int, obs: np.ndarray, base_reward: float) -> float:
        estimated_equity = float(np.clip(obs[0], 0, 1))
        shaping = base_reward
        if action == ACTION_FOLD and estimated_equity > self.equity_threshold_call:
            shaping -= 0.3
        elif action >= 2 and estimated_equity > self.equity_threshold_raise:
            shaping += 0.1 * estimated_equity
        return shaping

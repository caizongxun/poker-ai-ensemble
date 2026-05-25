import numpy as np
import random
from typing import List
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
_POT_IDX = 104   # pot / (stack_size * 2)
_STK_IDX = 105   # my stack / stack_size
_OPP_AF_IDX = 126  # opponent aggression factor (normalized 0~1)

# 当對手 AF 超過此閾値，認定對方是激進型，啟動 trap play
_AGGRO_THRESHOLD = 0.6


class StrategicAgent(BaseAgent):
    """
    策略型 Agent。
    新增：trap play
    - 對手是激進型（高 AF）時，強牌傼裝 check/call 引誘對方繼續 bluff raise
    - 對方 all-in 後呼叫且贏，給予額外獎勵（鼓勵學會認識 trap 機會）
    - 對手不激進時繼續用 pot odds + stack 深度決策
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
        pot_ratio = float(np.clip(obs[_POT_IDX], 0, 1))
        my_stack  = float(np.clip(obs[_STK_IDX], 0, 1))
        opp_af    = float(np.clip(
            obs[_OPP_AF_IDX] if len(obs) > _OPP_AF_IDX else 0.5, 0, 1))

        shaping = base_reward
        opp_is_aggro = opp_af > _AGGRO_THRESHOLD

        # ---- Trap play 模式（對手是激進型） ----
        if opp_is_aggro:
            # 對手激進時用 check/call 傼裝弱引誘 → 小 bonus（鼓勵 trap 姿態）
            if action == ACTION_CHECK_CALL:
                shaping += 0.08 * pot_ratio

            # 對手 all-in 後呼叫且贏 → 大 bonus（成功 trap）
            if action == ACTION_CHECK_CALL and base_reward > 0 and pot_ratio > 0.5:
                shaping += 0.25 * pot_ratio

            # 對手激進時還主動 fold → 懲罰（应該 trap 卻逃跑）
            if action == ACTION_FOLD and pot_ratio > 0.3:
                shaping -= 0.15 * pot_ratio

        # ---- 正常 pot odds 模式（對手不激進） ----
        else:
            # pot 很大還 fold → 懲罰
            if action == ACTION_FOLD and pot_ratio > 0.3:
                shaping -= 0.2 * pot_ratio

            # raise 時 stack 還很深 → 合理決策，小 bonus
            if action >= ACTION_RAISE_MIN and my_stack > 0.5:
                shaping += 0.05 * pot_ratio

        return shaping

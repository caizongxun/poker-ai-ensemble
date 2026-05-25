import random
import numpy as np
from typing import List, Tuple
from agents.base_agent import BaseAgent

ACTION_FOLD         = 0
ACTION_CHECK_CALL   = 1
ACTION_RAISE_MIN    = 2
ACTION_RAISE_50     = 3
ACTION_RAISE_75     = 4
ACTION_RAISE_POT    = 5
ACTION_RAISE_OVERBET = 6
ACTION_ALL_IN       = 7

# obs layout
_POT_IDX  = 104
_STK_IDX  = 105
_OPP_VPIP = 122
_OPP_PFR  = 123
_OPP_AF   = 124
_OPP_FOLD = 125
_OPP_WTSD = 126
_STREET_BASE = 107   # obs[107:111] = street one-hot

_AGGRO_THRESHOLD = 0.6
_EQUITY_RAISE    = 0.62
_EQUITY_CALL     = 0.38
_EQUITY_FOLD     = 0.25


def _estimate_equity(obs: np.ndarray, n_sim: int = 200) -> float:
    """
    快速 Monte Carlo equity 估算。
    利用 obs 的 hole-card one-hot 和 board one-hot 轉回牌索引，
    隨機發對手的兩張指定牌和剩餘公共牌，統計贏局比例。
    """
    try:
        from treys import Card, Evaluator
    except ImportError:
        return 0.5  # treys 不存在則回預設 50%

    RANKS = '23456789TJQKA'
    SUITS = 'cdhs'

    def to_treys(idx: int) -> int:
        return Card.new(RANKS[idx % 13] + SUITS[idx // 13])

    my_cards   = [i for i in range(52) if obs[i] > 0.5]
    board_cards = [i for i in range(52) if obs[52 + i] > 0.5]
    known       = set(my_cards + board_cards)
    remaining   = [i for i in range(52) if i not in known]

    if len(my_cards) < 2:
        return 0.5

    evaluator = Evaluator()
    wins = 0
    for _ in range(n_sim):
        sample = random.sample(remaining, 2 + max(0, 5 - len(board_cards)))
        opp_hole    = sample[:2]
        extra_board = sample[2:]
        full_board  = board_cards + extra_board
        if len(full_board) < 3:
            wins += random.random() > 0.5
            continue
        try:
            my_score  = evaluator.evaluate(
                [to_treys(c) for c in full_board],
                [to_treys(c) for c in my_cards])
            opp_score = evaluator.evaluate(
                [to_treys(c) for c in full_board],
                [to_treys(c) for c in opp_hole])
            if my_score < opp_score:
                wins += 1
            elif my_score == opp_score:
                wins += 0.5
        except Exception:
            wins += 0.5
    return wins / n_sim


class StrategicAgent(BaseAgent):
    """
    策略型 Agent。

    select_action 被覆寫：先用 NN 得到默認行動，
    再用 equity 估算修正最終行動。
    equity 計算耐時，只在 30% 的局里啟用。
    """

    def __init__(self, obs_size: int, action_size: int,
                 equity_override_prob: float = 0.30, **kwargs):
        super().__init__(obs_size, action_size, **kwargs)
        self.equity_override_prob = equity_override_prob

    def select_action(
        self, obs: np.ndarray, legal_actions: List[int]
    ) -> Tuple[int, float, float]:
        # 第一步：用 NN 得到基竜行動 + logp + value
        action, logp, value = super().select_action(obs, legal_actions)

        # 第二步：以 equity_override_prob 機率用 equity 修正
        if random.random() < self.equity_override_prob:
            equity = _estimate_equity(obs)
            pot    = float(obs[_POT_IDX]) if len(obs) > _POT_IDX else 0.5

            if equity >= _EQUITY_RAISE and len(legal_actions) > 2:
                # 高 equity -> 傾向加注：依局面大小選 pot 或 75%
                if pot > 0.5:
                    override = (ACTION_RAISE_POT
                                if ACTION_RAISE_POT in legal_actions
                                else ACTION_RAISE_75)
                else:
                    override = (ACTION_RAISE_75
                                if ACTION_RAISE_75 in legal_actions
                                else ACTION_RAISE_50)
                if override in legal_actions:
                    action = override

            elif equity <= _EQUITY_FOLD:
                # 低 equity -> fold
                if ACTION_FOLD in legal_actions:
                    action = ACTION_FOLD

            elif equity < _EQUITY_CALL:
                # 中段 equity -> check/call，不主動加注
                if ACTION_CHECK_CALL in legal_actions:
                    action = ACTION_CHECK_CALL

        return action, logp, value

    def compute_reward_shaping(
        self, action: int, obs: np.ndarray, base_reward: float
    ) -> float:
        pot_ratio = float(np.clip(obs[_POT_IDX], 0, 1)) if len(obs) > _POT_IDX else 0.5
        my_stack  = float(np.clip(obs[_STK_IDX], 0, 1)) if len(obs) > _STK_IDX else 0.5

        def _get(idx, default=0.5):
            return float(np.clip(obs[idx], 0, 1)) if len(obs) > idx else default

        opp_af   = _get(_OPP_AF)
        opp_fold = _get(_OPP_FOLD)
        shaping  = base_reward
        opp_is_aggro    = opp_af   > _AGGRO_THRESHOLD
        opp_folds_a_lot = opp_fold > 0.6

        if opp_is_aggro:
            if action == ACTION_CHECK_CALL:
                shaping += 0.08 * pot_ratio
            if action == ACTION_CHECK_CALL and base_reward > 0 and pot_ratio > 0.5:
                shaping += 0.25 * pot_ratio
            if action == ACTION_FOLD and pot_ratio > 0.3:
                shaping -= 0.15 * pot_ratio
        elif opp_folds_a_lot:
            if action >= ACTION_RAISE_MIN:
                shaping += 0.12 * pot_ratio
            if action == ACTION_CHECK_CALL and pot_ratio > 0.2:
                shaping -= 0.05 * pot_ratio
        else:
            if action == ACTION_FOLD and pot_ratio > 0.3:
                shaping -= 0.2 * pot_ratio
            if action >= ACTION_RAISE_MIN and my_stack > 0.5:
                shaping += 0.05 * pot_ratio

        return shaping

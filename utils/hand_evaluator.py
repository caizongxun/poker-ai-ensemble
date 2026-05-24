import random
from typing import List

try:
    from treys import Card, Evaluator
    TREYS_AVAILABLE = True
except ImportError:
    TREYS_AVAILABLE = False

RANKS = '23456789TJQKA'
SUITS = 'cdhs'


def index_to_card_str(idx: int) -> str:
    rank = RANKS[idx % 13]
    suit = SUITS[idx // 13]
    return rank + suit


def hand_strength(hole_cards: List[int], board: List[int]) -> float:
    """
    計算手牌強度 (0.0 ~ 1.0)，1.0 為最強。
    使用 treys library 評估。
    """
    if not TREYS_AVAILABLE or len(board) < 3:
        return 0.5
    evaluator = Evaluator()
    treys_hole = [Card.new(index_to_card_str(c)) for c in hole_cards]
    treys_board = [Card.new(index_to_card_str(c)) for c in board]
    score = evaluator.evaluate(treys_board, treys_hole)
    # treys score: 1 (Royal Flush) ~ 7462 (worst hand)
    return 1.0 - (score - 1) / 7461.0


def monte_carlo_equity(
    hole_cards: List[int],
    board: List[int],
    num_opponents: int = 1,
    simulations: int = 1000,
) -> float:
    """Monte Carlo 模擬估算 equity，回傳勝率 (0.0 ~ 1.0)。"""
    if not TREYS_AVAILABLE:
        return 0.5
    evaluator = Evaluator()
    deck = [i for i in range(52) if i not in hole_cards and i not in board]
    wins = 0
    needed_board = 5 - len(board)
    for _ in range(simulations):
        random.shuffle(deck)
        ptr = 0
        opp_hands = []
        for _ in range(num_opponents):
            opp_hands.append(deck[ptr: ptr + 2])
            ptr += 2
        extra_board = deck[ptr: ptr + needed_board]
        full_board = board + extra_board
        treys_board = [Card.new(index_to_card_str(c)) for c in full_board]
        my_score = evaluator.evaluate(
            treys_board,
            [Card.new(index_to_card_str(c)) for c in hole_cards]
        )
        best_opp = min(
            evaluator.evaluate(
                treys_board,
                [Card.new(index_to_card_str(c)) for c in opp]
            )
            for opp in opp_hands
        )
        if my_score < best_opp:
            wins += 1
    return wins / simulations


def pot_odds(call_amount: float, pot_size: float) -> float:
    """計算 pot odds，回傳需要的最低 equity。"""
    if call_amount <= 0:
        return 0.0
    return call_amount / (pot_size + call_amount)

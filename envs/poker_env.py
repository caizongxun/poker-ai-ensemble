import random
import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    from treys import Card, Evaluator, Deck
    TREYS_AVAILABLE = True
except ImportError:
    TREYS_AVAILABLE = False

# Abstract action indices
ACTION_FOLD = 0
ACTION_CHECK_CALL = 1
ACTION_RAISE_MIN = 2
ACTION_RAISE_50 = 3
ACTION_RAISE_75 = 4
ACTION_RAISE_POT = 5
ACTION_RAISE_OVERBET = 6
ACTION_ALL_IN = 7

STREET_PREFLOP = 0
STREET_FLOP = 1
STREET_TURN = 2
STREET_RIVER = 3

# Observation size: 52 (hole) + 52 (board) + 8 (pot/stack/street/position) + 10 (action history) + 5 (opp stats)
OBS_SIZE = 137


class NLHEState:
    """
    No-Limit Texas Hold'em 遊戲狀態，純 Python 實作。
    支援 2 人局 heads-up。
    """

    RANKS = '23456789TJQKA'
    SUITS = 'cdhs'

    def __init__(self, stack_size: int = 200, small_blind: int = 1, big_blind: int = 2):
        self.stack_size = stack_size
        self.sb = small_blind
        self.bb = big_blind
        self.reset()

    def reset(self):
        # 建牌並洗牌
        self.deck = list(range(52))
        random.shuffle(self.deck)
        self.deck_ptr = 0

        # 發牌
        self.hole_cards = [self._deal(2), self._deal(2)]  # [player0, player1]
        self.board = []

        # 籌碼
        self.stacks = [self.stack_size, self.stack_size]
        self.pot = 0
        self.bets = [0, 0]  # 當前辺投入金額

        # blinds
        self._post_blind(0, self.sb)
        self._post_blind(1, self.bb)

        self.street = STREET_PREFLOP
        self.current_player = 0  # preflop: SB 先行動
        self.last_aggressor = 1  # BB 最後加注
        self.num_actions_this_street = 0
        self.action_history = []  # (player, action_type, amount)
        self.terminal = False
        self.winner = -1
        self.opp_stats = [OpponentStats(), OpponentStats()]

    def _deal(self, n: int) -> List[int]:
        cards = self.deck[self.deck_ptr: self.deck_ptr + n]
        self.deck_ptr += n
        return cards

    def _post_blind(self, player: int, amount: int):
        actual = min(amount, self.stacks[player])
        self.stacks[player] -= actual
        self.bets[player] += actual
        self.pot += actual

    def get_legal_actions(self) -> List[int]:
        if self.terminal:
            return []
        actions = [ACTION_FOLD, ACTION_CHECK_CALL]
        call_amount = self.bets[1 - self.current_player] - self.bets[self.current_player]
        if call_amount < 0:
            call_amount = 0
        if self.stacks[self.current_player] > call_amount:
            actions += [ACTION_RAISE_MIN, ACTION_RAISE_50, ACTION_RAISE_75,
                        ACTION_RAISE_POT, ACTION_RAISE_OVERBET, ACTION_ALL_IN]
        return actions

    def apply_action(self, abstract_action: int):
        player = self.current_player
        call_amount = max(0, self.bets[1 - player] - self.bets[player])
        pot_before = self.pot

        if abstract_action == ACTION_FOLD:
            self.terminal = True
            self.winner = 1 - player
            self.action_history.append((player, 'fold', 0))
            self.opp_stats[player].update(0, self.street, pot_before)

        elif abstract_action == ACTION_CHECK_CALL:
            actual = min(call_amount, self.stacks[player])
            self.stacks[player] -= actual
            self.bets[player] += actual
            self.pot += actual
            self.action_history.append((player, 'call', actual))
            self.opp_stats[player].update(1, self.street, pot_before, actual)
            self.num_actions_this_street += 1
            self._maybe_advance_street()

        else:
            # raise 系列
            raise_ratios = {
                ACTION_RAISE_MIN: 0.0,
                ACTION_RAISE_50: 0.5,
                ACTION_RAISE_75: 0.75,
                ACTION_RAISE_POT: 1.0,
                ACTION_RAISE_OVERBET: 1.5,
                ACTION_ALL_IN: 99.0,
            }
            ratio = raise_ratios.get(abstract_action, 1.0)
            if abstract_action == ACTION_ALL_IN:
                raise_total = self.stacks[player] + self.bets[player]
            else:
                raise_size = max(self.bb, int(ratio * self.pot))
                raise_total = self.bets[1 - player] + raise_size
                raise_total = min(raise_total, self.stacks[player] + self.bets[player])

            additional = raise_total - self.bets[player]
            additional = min(additional, self.stacks[player])
            self.stacks[player] -= additional
            self.bets[player] += additional
            self.pot += additional
            self.last_aggressor = player
            self.action_history.append((player, 'raise', additional))
            self.opp_stats[player].update(2, self.street, pot_before, additional)
            self.num_actions_this_street = 1
            self._switch_player()

    def _maybe_advance_street(self):
        call_amount = abs(self.bets[0] - self.bets[1])
        both_acted = self.num_actions_this_street >= 2
        bets_equal = call_amount == 0

        if both_acted and bets_equal:
            if self.street == STREET_RIVER:
                self._showdown()
            else:
                self.street += 1
                if self.street == STREET_FLOP:
                    self.board += self._deal(3)
                elif self.street == STREET_TURN:
                    self.board += self._deal(1)
                elif self.street == STREET_RIVER:
                    self.board += self._deal(1)
                self.bets = [0, 0]
                self.num_actions_this_street = 0
                self.current_player = 1  # OOP goes first post-flop
        else:
            self._switch_player()

    def _switch_player(self):
        self.current_player = 1 - self.current_player

    def _showdown(self):
        self.terminal = True
        if TREYS_AVAILABLE and len(self.board) >= 3:
            evaluator = Evaluator()
            RANKS = '23456789TJQKA'
            SUITS = 'cdhs'
            def to_treys(idx):
                return Card.new(RANKS[idx % 13] + SUITS[idx // 13])
            board_t = [to_treys(c) for c in self.board]
            scores = [
                evaluator.evaluate(board_t, [to_treys(c) for c in self.hole_cards[i]])
                for i in range(2)
            ]
            self.winner = 0 if scores[0] < scores[1] else 1  # lower = better in treys
        else:
            self.winner = random.randint(0, 1)  # fallback

    def get_reward(self, player: int) -> float:
        if not self.terminal:
            return 0.0
        if self.winner == player:
            return float(self.pot - (self.stack_size - self.stacks[player]))
        else:
            return float(-(self.stack_size - self.stacks[player]))

    def encode_obs(self, player: int) -> np.ndarray:
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        # hole cards one-hot (52)
        for c in self.hole_cards[player]:
            obs[c] = 1.0
        # board one-hot (52)
        for c in self.board:
            obs[52 + c] = 1.0
        # pot ratio
        obs[104] = self.pot / (self.stack_size * 2)
        # stack ratio
        obs[105] = self.stacks[player] / self.stack_size
        obs[106] = self.stacks[1 - player] / self.stack_size
        # street one-hot
        obs[107 + self.street] = 1.0
        # position
        obs[111] = float(player)
        # SPR (Stack to Pot Ratio)
        obs[112] = min(self.stacks[player] / max(self.pot, 1), 10.0) / 10.0
        # last 3 actions encoding
        for i, (p, atype, amt) in enumerate(self.action_history[-3:]):
            base = 113 + i * 3
            obs[base] = float(p)
            obs[base + 1] = ['fold','call','raise'].index(atype) / 2.0 if atype in ['fold','call','raise'] else 0
            obs[base + 2] = min(amt / max(self.pot, 1), 3.0) / 3.0
        # opp stats
        opp_vec = self.opp_stats[1 - player].to_vector()
        obs[122: 122 + len(opp_vec)] = opp_vec
        return obs


class PokerEnv:
    """
    No-Limit Texas Hold'em 環境。
    純 Python 實作，不依賴 OpenSpiel，支援 Windows。
    """

    def __init__(self, num_players: int = 2, stack_size: int = 200):
        assert num_players == 2, "Currently supports heads-up (2 players) only"
        self.num_players = num_players
        self.stack_size = stack_size
        self.state: Optional[NLHEState] = None

    def reset(self) -> np.ndarray:
        self.state = NLHEState(stack_size=self.stack_size)
        return self.state.encode_obs(self.state.current_player)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        player = self.state.current_player
        legal = self.get_legal_actions()
        if action not in legal:
            action = ACTION_CHECK_CALL if ACTION_CHECK_CALL in legal else legal[0]
        self.state.apply_action(action)
        done = self.state.terminal
        if done:
            reward = self.state.get_reward(player)
            obs = np.zeros(OBS_SIZE, dtype=np.float32)
        else:
            reward = 0.0
            obs = self.state.encode_obs(self.state.current_player)
        return obs, reward, done, {}

    def get_legal_actions(self) -> List[int]:
        return self.state.get_legal_actions()

    @property
    def observation_size(self) -> int:
        return OBS_SIZE

    @property
    def action_size(self) -> int:
        return 8


class OpponentStats:
    """追蹤對手的行為統計，供 Strategic/Deceptive Agent 使用。"""

    def __init__(self):
        self._hands = 0
        self._vpip = 0
        self._pfr = 0
        self._agg = 0
        self._passive = 0
        self._fold_cbet = 0
        self._cbet_faced = 0
        self._wtsd = 0
        self._showdowns = 0

    def update(self, action: int, street: int, pot_size: float, bet_size: float = 0.0):
        if street == STREET_PREFLOP:
            self._hands += 1
            if action != 0:
                self._vpip += 1
            if action >= 2:
                self._pfr += 1
        if action >= 2:
            self._agg += 1
        elif action == 1:
            self._passive += 1

    def to_vector(self) -> np.ndarray:
        return np.array([
            self._vpip / max(self._hands, 1),
            self._pfr / max(self._hands, 1),
            min(self._agg / max(self._passive, 1), 5.0) / 5.0,
            self._fold_cbet / max(self._cbet_faced, 1),
            self._wtsd / max(self._showdowns, 1),
        ], dtype=np.float32)

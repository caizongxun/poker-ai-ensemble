import numpy as np
import pyspiel
from typing import Dict, List, Tuple

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


class PokerEnv:
    """
    OpenSpiel Texas Hold'em 環境包裝。
    提供統一的 state encoding 供所有子模型使用。
    """

    def __init__(self, num_players: int = 2, stack_size: int = 200):
        self.num_players = num_players
        self.stack_size = stack_size
        self.game = pyspiel.load_game(
            "universal_poker",
            {
                "betting": "nolimit",
                "numPlayers": num_players,
                "numRounds": 4,
                "blind": "1 2",
                "firstPlayer": "1 1 1 1",
                "numSuits": 4,
                "numRanks": 13,
                "numHoleCards": 2,
                "numBoardCards": "0 3 1 1",
                "stack": " ".join([str(stack_size)] * num_players),
            },
        )
        self.state = None

    def reset(self) -> np.ndarray:
        self.state = self.game.new_initial_state()
        while self.state.is_chance_node():
            outcomes = self.state.chance_outcomes()
            action = np.random.choice(
                [o[0] for o in outcomes],
                p=[o[1] for o in outcomes]
            )
            self.state.apply_action(action)
        return self._encode_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        legal_actions = self.get_legal_actions()
        mapped_action = self._map_action(action, legal_actions)
        self.state.apply_action(mapped_action)

        while self.state.is_chance_node() and not self.state.is_terminal():
            outcomes = self.state.chance_outcomes()
            a = np.random.choice(
                [o[0] for o in outcomes],
                p=[o[1] for o in outcomes]
            )
            self.state.apply_action(a)

        done = self.state.is_terminal()
        reward = (
            self.state.returns()[self.state.current_player()]
            if done else 0.0
        )
        obs = self._encode_state() if not done else np.zeros(self.observation_size)
        return obs, reward, done, {}

    def get_legal_actions(self) -> List[int]:
        return self.state.legal_actions()

    def _map_action(self, abstract_action: int, legal_actions: List[int]) -> int:
        if abstract_action == ACTION_FOLD and 0 in legal_actions:
            return 0
        if abstract_action == ACTION_CHECK_CALL and 1 in legal_actions:
            return 1
        pot = self.stack_size * 0.1  # simplified pot estimate
        target_ratios = {
            ACTION_RAISE_MIN: 0.0,
            ACTION_RAISE_50: 0.5,
            ACTION_RAISE_75: 0.75,
            ACTION_RAISE_POT: 1.0,
            ACTION_RAISE_OVERBET: 1.5,
            ACTION_ALL_IN: 99.0,
        }
        target = target_ratios.get(abstract_action, 1.0) * pot
        candidates = [a for a in legal_actions if a >= 2]
        if candidates:
            return min(candidates, key=lambda x: abs(x - target))
        return legal_actions[-1]

    def _encode_state(self) -> np.ndarray:
        player = self.state.current_player()
        obs = self.state.observation_tensor(player)
        return np.array(obs, dtype=np.float32)

    @property
    def observation_size(self) -> int:
        return self.game.observation_tensor_size()

    @property
    def action_size(self) -> int:
        return 8


class OpponentStats:
    """追蹤對手的行為統計，供 Strategic/Deceptive Agent 使用。"""

    def __init__(self):
        self.hands_seen = 0
        self.vpip_count = 0
        self.pfr_count = 0
        self.aggression_actions = 0
        self.passive_actions = 0
        self.fold_to_cbet = 0
        self.cbet_faced = 0
        self.wtsd = 0
        self.showdowns = 0

    @property
    def vpip(self) -> float:
        return self.vpip_count / max(self.hands_seen, 1)

    @property
    def pfr(self) -> float:
        return self.pfr_count / max(self.hands_seen, 1)

    @property
    def af(self) -> float:
        return self.aggression_actions / max(self.passive_actions, 1)

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.vpip,
            self.pfr,
            min(self.af, 5.0) / 5.0,
            self.fold_to_cbet / max(self.cbet_faced, 1),
            self.wtsd / max(self.showdowns, 1),
        ], dtype=np.float32)

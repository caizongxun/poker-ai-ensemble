import numpy as np
from collections import deque
from typing import Dict


class OpponentModel:
    """
    線上對手行為建模。
    維護每個對手的行為統計，提供特徵向量給 Strategic/Deceptive Agent。
    """

    def __init__(self, player_id: int, history_len: int = 200):
        self.player_id = player_id
        self.action_history = deque(maxlen=history_len)
        self._hands = 0
        self._vpip = 0
        self._pfr = 0
        self._agg = 0
        self._passive = 0
        self._fold_cbet = 0
        self._cbet_faced = 0
        self._wtsd = 0
        self._showdowns = 0
        self._bet_sizes = deque(maxlen=50)

    def update(self, action: int, street: int, pot_size: float, bet_size: float = 0.0):
        """
        每次對手行動後呼叫，更新統計。
        action: 0=fold, 1=check/call, 2+=raise
        """
        self.action_history.append((action, street, pot_size, bet_size))
        if street == 0:  # preflop
            self._hands += 1
            if action != 0:
                self._vpip += 1
            if action >= 2:
                self._pfr += 1
        if action >= 2:
            self._agg += 1
        elif action == 1:
            self._passive += 1
        if bet_size > 0 and pot_size > 0:
            self._bet_sizes.append(bet_size / pot_size)

    def classify_type(self) -> str:
        """依統計數據分類對手類型。"""
        vpip = self._vpip / max(self._hands, 1)
        pfr = self._pfr / max(self._hands, 1)
        af = self._agg / max(self._passive, 1)
        if vpip < 0.2 and pfr < 0.15:
            return "nit"
        elif vpip < 0.3 and pfr < 0.2:
            return "tight_passive"
        elif vpip > 0.4 and af > 2.0:
            return "loose_aggressive"
        elif vpip > 0.4 and af < 1.5:
            return "loose_passive"
        elif vpip < 0.3 and af > 2.0:
            return "tight_aggressive"
        return "unknown"

    def to_vector(self) -> np.ndarray:
        avg_bet = float(np.mean(self._bet_sizes)) if self._bet_sizes else 0.5
        return np.array([
            self._vpip / max(self._hands, 1),
            self._pfr / max(self._hands, 1),
            min(self._agg / max(self._passive, 1), 5.0) / 5.0,
            self._fold_cbet / max(self._cbet_faced, 1),
            self._wtsd / max(self._showdowns, 1),
            avg_bet,
        ], dtype=np.float32)

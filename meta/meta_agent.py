import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple

# 對手風格 embedding 維度：[vpip, pfr, af, fold_freq]
OPP_STYLE_DIM = 4


class MoEGatingNetwork(nn.Module):
    """
    Gating Network 帶對手風格 embedding。
    Input dim = obs_size + opp_style_dim
    """

    def __init__(self, obs_size: int, num_experts: int = 4,
                 hidden_size: int = 256,
                 opp_style_dim: int = OPP_STYLE_DIM):
        super().__init__()
        self.num_experts   = num_experts
        self.opp_style_dim = opp_style_dim
        self.total_input   = obs_size + opp_style_dim   # 明確記錄以便 debug

        self.net = nn.Sequential(
            nn.Linear(self.total_input, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts),
        )
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)

    def forward(self, x: torch.Tensor,
                opp_style: torch.Tensor = None) -> torch.Tensor:
        if opp_style is None:
            opp_style = torch.zeros(
                x.shape[0], self.opp_style_dim, device=x.device)
        # 必須確認兩者 batch dim 對齊
        if opp_style.shape[0] != x.shape[0]:
            opp_style = opp_style.expand(x.shape[0], -1)
        combined = torch.cat([x, opp_style], dim=-1)
        assert combined.shape[-1] == self.total_input, (
            f"Gating input mismatch: expected {self.total_input}, "
            f"got {combined.shape[-1]}. "
            f"obs={x.shape[-1]}, opp_style={opp_style.shape[-1]}"
        )
        logits = self.net(combined)
        temp   = self.temperature.abs().clamp(min=0.5, max=5.0)
        return torch.softmax(logits / temp, dim=-1)

    def entropy_loss(self, weights: torch.Tensor) -> torch.Tensor:
        entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()
        return -entropy


class OpponentStyleTracker:
    """
    在線追蹤對手打牌風格，產生 4-dim embedding。
    vpip  : 入局率
    pfr   : preflop raise 率
    af    : aggression factor 歸一化
    fold  : fold-to-raise 率
    """

    def __init__(self, window: int = 50):
        self.window = window
        self._hand_count  = 0
        self._vpip_count  = 0
        self._pfr_count   = 0
        self._raise_count = 0
        self._call_count  = 0
        self._fold_to_raise  = 0
        self._faced_raise    = 0

    def update(self, action: int, street: int, faced_raise: bool = False):
        if street == 0:
            self._hand_count += 1
            if action >= 1:
                self._vpip_count += 1
            if action >= 2:
                self._pfr_count += 1
        if action >= 2:
            self._raise_count += 1
        elif action == 1:
            self._call_count += 1
        if faced_raise:
            self._faced_raise += 1
            if action == 0:
                self._fold_to_raise += 1

    def get_embedding(self) -> np.ndarray:
        h    = max(self._hand_count, 1)
        denom = max(self._raise_count + self._call_count, 1)
        vpip = min(self._vpip_count / h, 1.0)
        pfr  = min(self._pfr_count  / h, 1.0)
        af   = min(self._raise_count / denom, 1.0)
        fold = self._fold_to_raise / max(self._faced_raise, 1)
        return np.array([vpip, pfr, af, fold], dtype=np.float32)

    def reset(self):
        self.__init__(self.window)


class MetaAgent:
    """
    Meta Agent: Mixture of Experts.
    修正： MetaAgent.__init__ 明確將 obs_size 傳入 MoEGatingNetwork，
            gating 內部自己加 opp_style_dim，就不會再有 input size 不對齊。
    """

    EXPERT_NAMES = ["aggressive", "conservative", "strategic", "deceptive"]

    def __init__(self, obs_size: int, action_size: int, sub_agents: Dict,
                 mode: str = "soft", lr: float = 1e-4,
                 entropy_coef: float = 0.05, device: str = "cpu"):
        self.device        = torch.device(device)
        self.mode          = mode
        self.action_size   = action_size
        self.sub_agents    = sub_agents
        self.entropy_coef  = entropy_coef
        self.opp_tracker   = OpponentStyleTracker()

        # 傳 obs_size 即可，MoEGatingNetwork 內部自動加 OPP_STYLE_DIM
        self.gating = MoEGatingNetwork(obs_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.gating.parameters(), lr=lr)

        self.value_net = nn.Sequential(
            nn.Linear(obs_size, 256), nn.ReLU(), nn.Linear(256, 1)
        ).to(self.device)
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=lr)

    def _opp_tensor(self, batch_size: int = 1) -> torch.Tensor:
        emb = self.opp_tracker.get_embedding()
        t   = torch.FloatTensor(emb).unsqueeze(0).to(self.device)
        return t.expand(batch_size, -1)

    def observe_opponent(self, action: int, street: int,
                         faced_raise: bool = False):
        self.opp_tracker.update(action, street, faced_raise)

    def reset_opponent_tracker(self):
        self.opp_tracker.reset()

    def select_action(
        self, obs: np.ndarray, legal_actions: List[int]
    ) -> Tuple[int, float, float, np.ndarray]:
        x          = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        opp_style  = self._opp_tensor(1)
        with torch.no_grad():
            weights = self.gating(x, opp_style).squeeze(0)  # [4]

        expert_probs = []
        for name in self.EXPERT_NAMES:
            p = self.sub_agents[name].get_action_probs(obs, legal_actions)
            expert_probs.append(torch.FloatTensor(p).to(self.device))
        expert_probs = torch.stack(expert_probs)  # [4, action_size]

        if self.mode == "soft":
            mixed = (weights.unsqueeze(1) * expert_probs).sum(0)
            mixed = mixed.clamp(min=1e-8)
            mixed = mixed / mixed.sum()
            dist  = torch.distributions.Categorical(probs=mixed)
        else:
            dist = torch.distributions.Categorical(
                probs=expert_probs[weights.argmax()])

        action   = dist.sample()
        log_prob = dist.log_prob(action)
        with torch.no_grad():
            value = self.value_net(x).squeeze()

        return (action.item(), log_prob.item(),
                value.item(), weights.detach().cpu().numpy())

    def get_gating_weights(self, obs: np.ndarray) -> np.ndarray:
        x         = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        opp_style = self._opp_tensor(1)
        with torch.no_grad():
            w = self.gating(x, opp_style).squeeze(0)
        return w.cpu().numpy()

    def save(self, path: str):
        torch.save({
            "gating": self.gating.state_dict(),
            "value":  self.value_net.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.gating.load_state_dict(ckpt["gating"])
        self.value_net.load_state_dict(ckpt["value"])

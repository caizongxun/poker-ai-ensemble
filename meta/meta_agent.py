import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple

# 對手風格 embedding 維度：[vpip, pfr, af, fold_freq] 各 normalized 0~1
OPP_STYLE_DIM = 4


class MoEGatingNetwork(nn.Module):
    """
    Gating Network 帶對手風格 embedding。
    Input = obs + opp_style_embedding
    這樣 gating 才能真正學會「面對此類對手用哪個專家」。
    """

    def __init__(self, obs_size: int, num_experts: int = 4,
                 hidden_size: int = 256, opp_style_dim: int = OPP_STYLE_DIM):
        super().__init__()
        self.num_experts = num_experts
        self.opp_style_dim = opp_style_dim
        total_input = obs_size + opp_style_dim

        self.net = nn.Sequential(
            nn.Linear(total_input, hidden_size),
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
        combined = torch.cat([x, opp_style], dim=-1)
        logits = self.net(combined)
        temp = self.temperature.abs().clamp(min=0.5, max=5.0)
        return torch.softmax(logits / temp, dim=-1)

    def entropy_loss(self, weights: torch.Tensor) -> torch.Tensor:
        entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()
        return -entropy


class OpponentStyleTracker:
    """
    在線追蹤對手打牌風格，產生 4 維 embedding。
    vpip  : 入局率 (call/raise preflop)
    pfr   : preflop raise 率
    af    : aggression factor = (raise+bet) / call
    fold  : 遇到 raise 的 fold 率
    """

    def __init__(self, window: int = 50):
        self.window = window
        self._actions: List[int] = []   # 0=fold,1=call,2+=raise
        self._vpip_count = 0
        self._pfr_count = 0
        self._hand_count = 0
        self._raise_count = 0
        self._call_count = 0
        self._fold_to_raise = 0
        self._faced_raise = 0

    def update(self, action: int, street: int, faced_raise: bool = False):
        self._actions.append(action)
        if len(self._actions) > self.window:
            self._actions.pop(0)
        if street == 0:  # preflop
            self._hand_count += 1
            if action >= 1:   # call or raise
                self._vpip_count += 1
            if action >= 2:   # raise
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
        h = max(self._hand_count, 1)
        denom = max(self._raise_count + self._call_count, 1)
        vpip = min(self._vpip_count / h, 1.0)
        pfr  = min(self._pfr_count / h, 1.0)
        af   = min(self._raise_count / denom, 1.0)
        fold = self._fold_to_raise / max(self._faced_raise, 1)
        return np.array([vpip, pfr, af, fold], dtype=np.float32)

    def reset(self):
        self.__init__(self.window)


class MetaAgent:
    """
    Meta Agent: Mixture of Experts 整合四個子模型。
    修正: gating input 加入對手風格 embedding，
            讓 meta 真正學到「看對手类型切換策略」。
    """

    EXPERT_NAMES = ["aggressive", "conservative", "strategic", "deceptive"]

    def __init__(self, obs_size: int, action_size: int, sub_agents: Dict,
                 mode: str = "soft", lr: float = 1e-4,
                 entropy_coef: float = 0.05, device: str = "cpu"):
        self.device = torch.device(device)
        self.mode = mode
        self.action_size = action_size
        self.sub_agents = sub_agents
        self.entropy_coef = entropy_coef
        self.opp_tracker = OpponentStyleTracker()

        self.gating = MoEGatingNetwork(obs_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.gating.parameters(), lr=lr)

        self.value_net = nn.Sequential(
            nn.Linear(obs_size, 256), nn.ReLU(), nn.Linear(256, 1)
        ).to(self.device)
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=lr)

    def _get_opp_style_tensor(self) -> torch.Tensor:
        emb = self.opp_tracker.get_embedding()
        return torch.FloatTensor(emb).unsqueeze(0).to(self.device)

    def observe_opponent(self, action: int, street: int,
                         faced_raise: bool = False):
        """每次對手行動後呼叫，更新對手風格 tracker。"""
        self.opp_tracker.update(action, street, faced_raise)

    def reset_opponent_tracker(self):
        self.opp_tracker.reset()

    def select_action(
        self, obs: np.ndarray, legal_actions: List[int]
    ) -> Tuple[int, float, float, np.ndarray]:
        x = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        opp_style = self._get_opp_style_tensor()
        with torch.no_grad():
            weights = self.gating(x, opp_style).squeeze(0)

        expert_probs = []
        for name in self.EXPERT_NAMES:
            probs = self.sub_agents[name].get_action_probs(obs, legal_actions)
            expert_probs.append(torch.FloatTensor(probs).to(self.device))
        expert_probs = torch.stack(expert_probs)

        if self.mode == "soft":
            mixed_probs = (weights.unsqueeze(1) * expert_probs).sum(dim=0)
            mixed_probs = mixed_probs.clamp(min=1e-8)
            mixed_probs = mixed_probs / mixed_probs.sum()
            dist = torch.distributions.Categorical(probs=mixed_probs)
        else:
            best_expert = weights.argmax().item()
            dist = torch.distributions.Categorical(
                probs=expert_probs[best_expert])

        action = dist.sample()
        log_prob = dist.log_prob(action)
        with torch.no_grad():
            value = self.value_net(x).squeeze()

        return (action.item(), log_prob.item(),
                value.item(), weights.detach().cpu().numpy())

    def get_gating_weights(self, obs: np.ndarray) -> np.ndarray:
        x = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        opp_style = self._get_opp_style_tensor()
        with torch.no_grad():
            weights = self.gating(x, opp_style).squeeze(0)
        return weights.cpu().numpy()

    def save(self, path: str):
        torch.save({
            "gating": self.gating.state_dict(),
            "value": self.value_net.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.gating.load_state_dict(ckpt["gating"])
        self.value_net.load_state_dict(ckpt["value"])

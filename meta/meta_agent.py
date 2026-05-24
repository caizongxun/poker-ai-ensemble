import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple


class MoEGatingNetwork(nn.Module):
    """
    Mixture of Experts Gating Network。
    加入 entropy regularization 防止 collapse 到單一 expert。
    """

    def __init__(self, obs_size: int, num_experts: int = 4, hidden_size: int = 256):
        super().__init__()
        self.num_experts = num_experts
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts),
        )
        # temperature 初始值設高一點，讓初期 weights 較均勻
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        # clamp temperature 避免過度收斂
        temp = self.temperature.abs().clamp(min=0.5, max=5.0)
        weights = torch.softmax(logits / temp, dim=-1)
        return weights

    def entropy_loss(self, weights: torch.Tensor) -> torch.Tensor:
        """鼓勵 weights 保持多樣性，防止 collapse。"""
        entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()
        return -entropy  # 最小化負 entropy = 最大化 entropy


class MetaAgent:
    """
    Meta Agent：整合四個子模型的 Mixture of Experts。
    修正：select_action 回傳 4-tuple (action, log_prob, value, weights)。
    """

    EXPERT_NAMES = ["aggressive", "conservative", "strategic", "deceptive"]

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        sub_agents: Dict,
        mode: str = "soft",
        lr: float = 1e-4,
        entropy_coef: float = 0.05,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.mode = mode
        self.action_size = action_size
        self.sub_agents = sub_agents
        self.entropy_coef = entropy_coef

        self.gating = MoEGatingNetwork(obs_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.gating.parameters(), lr=lr)

        self.value_net = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ).to(self.device)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)

    def select_action(
        self, obs: np.ndarray, legal_actions: List[int]
    ) -> Tuple[int, float, float, np.ndarray]:
        """
        回傳 (action, log_prob, value, weights)
        weights: shape [4]，各 expert 的 gating weight
        """
        x = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            weights = self.gating(x).squeeze(0)  # [4]

        expert_probs = []
        for name in self.EXPERT_NAMES:
            probs = self.sub_agents[name].get_action_probs(obs, legal_actions)
            expert_probs.append(torch.FloatTensor(probs).to(self.device))
        expert_probs = torch.stack(expert_probs)  # [4, action_size]

        if self.mode == "soft":
            mixed_probs = (weights.unsqueeze(1) * expert_probs).sum(dim=0)
            # clamp 防止數值問題
            mixed_probs = mixed_probs.clamp(min=1e-8)
            mixed_probs = mixed_probs / mixed_probs.sum()
            dist = torch.distributions.Categorical(probs=mixed_probs)
        else:
            best_expert = weights.argmax().item()
            dist = torch.distributions.Categorical(probs=expert_probs[best_expert])

        action = dist.sample()
        log_prob = dist.log_prob(action)

        with torch.no_grad():
            value = self.value_net(x).squeeze()

        return (
            action.item(),
            log_prob.item(),
            value.item(),
            weights.detach().cpu().numpy(),
        )

    def update(self, obs: np.ndarray, action: int, reward: float,
               log_prob_old: float, value_old: float) -> Dict:
        """單步 PPO-like update，含 entropy regularization 防止 gating collapse。"""
        x = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        weights = self.gating(x).squeeze(0)

        expert_probs = []
        for name in self.EXPERT_NAMES:
            probs = self.sub_agents[name].get_action_probs(obs, list(range(self.action_size)))
            expert_probs.append(torch.FloatTensor(probs).to(self.device))
        expert_probs = torch.stack(expert_probs)

        mixed_probs = (weights.unsqueeze(1) * expert_probs).sum(dim=0).clamp(min=1e-8)
        mixed_probs = mixed_probs / mixed_probs.sum()
        dist = torch.distributions.Categorical(probs=mixed_probs)

        log_prob = dist.log_prob(torch.tensor(action).to(self.device))
        value = self.value_net(x).squeeze()

        advantage = reward - value.item()
        policy_loss = -log_prob * advantage
        value_loss = (value - reward) ** 2
        entropy_loss = self.entropy_coef * self.gating.entropy_loss(weights.unsqueeze(0))

        total_loss = policy_loss + 0.5 * value_loss + entropy_loss

        self.optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.gating.parameters(), 0.5)
        self.optimizer.step()
        self.value_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }

    def get_gating_weights(self, obs: np.ndarray) -> np.ndarray:
        x = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            weights = self.gating(x).squeeze(0)
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

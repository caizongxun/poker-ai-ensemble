import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple


class MoEGatingNetwork(nn.Module):
    """
    Mixture of Experts Gating Network。
    輸入: 當前局面 obs
    輸出: 4 維 weight vector [w_aggressive, w_conservative, w_strategic, w_deceptive]
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
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        weights = torch.softmax(logits / self.temperature.abs().clamp(min=0.1), dim=-1)
        return weights


class MetaAgent:
    """
    Meta Agent：整合四個子模型的 Mixture of Experts。

    方案 A (soft MoE)：
        final_action_probs = softmax(w) · [aggressive_π, conservative_π, strategic_π, deceptive_π]

    方案 B (hard selection)：
        選擇 weight 最高的子模型主導本輪決策。
    """

    EXPERT_NAMES = ["aggressive", "conservative", "strategic", "deceptive"]

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        sub_agents: Dict,
        mode: str = "soft",
        lr: float = 1e-4,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.mode = mode
        self.action_size = action_size
        self.sub_agents = sub_agents

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
        x = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        weights = self.gating(x).squeeze(0)  # [4]

        expert_probs = []
        for name in self.EXPERT_NAMES:
            probs = self.sub_agents[name].get_action_probs(obs, legal_actions)
            expert_probs.append(torch.FloatTensor(probs))
        expert_probs = torch.stack(expert_probs)  # [4, action_size]

        if self.mode == "soft":
            mixed_probs = (weights.unsqueeze(1) * expert_probs).sum(dim=0)
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

    def get_gating_weights(self, obs: np.ndarray) -> np.ndarray:
        """查詢當前局面下各子模型的權重分配。"""
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
        ckpt = torch.load(path, map_location=self.device)
        self.gating.load_state_dict(ckpt["gating"])
        self.value_net.load_state_dict(ckpt["value"])

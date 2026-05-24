import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple


class ActorCritic(nn.Module):
    """共用的 Actor-Critic 網路骨架，所有子模型繼承此類。"""

    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.shared(x)
        return self.actor(feat), self.critic(feat)

    def get_action_distribution(self, x: torch.Tensor, legal_mask: torch.Tensor = None):
        logits, value = self.forward(x)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask.bool(), float('-inf'))
        dist = torch.distributions.Categorical(logits=logits)
        return dist, value


class BaseAgent(ABC):
    """
    子模型基底類別。
    各風格子模型覆寫 compute_reward_shaping() 加入風格偏置。
    """

    def __init__(self, obs_size: int, action_size: int, lr: float = 3e-4, device: str = "cpu"):
        self.device = torch.device(device)
        self.network = ActorCritic(obs_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.action_size = action_size

    def select_action(self, obs: np.ndarray, legal_actions: List[int]) -> Tuple[int, float, float]:
        x = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mask = torch.zeros(self.action_size)
        for a in legal_actions:
            if a < self.action_size:
                mask[a] = 1
        dist, value = self.network.get_action_distribution(x, mask)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def get_action_probs(self, obs: np.ndarray, legal_actions: List[int]) -> np.ndarray:
        """回傳完整 action probability distribution，供 Meta Agent 使用。"""
        x = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mask = torch.zeros(self.action_size)
        for a in legal_actions:
            if a < self.action_size:
                mask[a] = 1
        with torch.no_grad():
            logits, _ = self.network(x)
            logits = logits.masked_fill(~mask.bool(), float('-inf'))
            probs = torch.softmax(logits, dim=-1)
        return probs.squeeze(0).cpu().numpy()

    @abstractmethod
    def compute_reward_shaping(self, action: int, obs: np.ndarray, base_reward: float) -> float:
        """各風格子模型覆寫此方法，加入風格相關的 reward shaping。"""
        pass

    def save(self, path: str):
        torch.save(self.network.state_dict(), path)

    def load(self, path: str):
        self.network.load_state_dict(torch.load(path, map_location=self.device))

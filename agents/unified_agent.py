import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

OPP_STYLE_DIM = 4  # [vpip, pfr, af, fold_freq]


class OpponentStyleTracker:
    """Online tracker producing 4-dim opponent style embedding."""

    def __init__(self, window: int = 80):
        self.window = window
        self._hands = 0
        self._vpip  = 0
        self._pfr   = 0
        self._raise = 0
        self._call  = 0
        self._fold_to_raise = 0
        self._faced_raise   = 0

    def update(self, action: int, street: int, faced_raise: bool = False):
        if street == 0:
            self._hands += 1
            if action >= 1: self._vpip += 1
            if action >= 2: self._pfr  += 1
        if action >= 2: self._raise += 1
        elif action == 1: self._call += 1
        if faced_raise:
            self._faced_raise += 1
            if action == 0: self._fold_to_raise += 1

    def embedding(self) -> np.ndarray:
        h    = max(self._hands, 1)
        denom = max(self._raise + self._call, 1)
        vpip = min(self._vpip  / h,     1.0)
        pfr  = min(self._pfr   / h,     1.0)
        af   = min(self._raise / denom, 1.0)
        fold = self._fold_to_raise / max(self._faced_raise, 1)
        return np.array([vpip, pfr, af, fold], dtype=np.float32)

    def reset(self):
        self.__init__(self.window)


class UnifiedNetwork(nn.Module):
    """
    Single Actor-Critic that takes (obs, opp_style) as input.
    Hidden size 512 to absorb capacity from all 4 sub-agents.
    """

    def __init__(self, obs_size: int, action_size: int,
                 hidden_size: int = 512):
        super().__init__()
        total_in = obs_size + OPP_STYLE_DIM

        self.shared = nn.Sequential(
            nn.Linear(total_in, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, action_size),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, obs: torch.Tensor,
                opp_style: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x    = torch.cat([obs, opp_style], dim=-1)
        feat = self.shared(x)
        return self.actor(feat), self.critic(feat)

    def get_dist_and_value(self, obs: torch.Tensor,
                           opp_style: torch.Tensor,
                           legal_mask: torch.Tensor = None):
        logits, value = self.forward(obs, opp_style)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask.bool(), float('-inf'))
        dist = torch.distributions.Categorical(logits=logits)
        return dist, value


class UnifiedAgent:
    """
    Single unified poker agent.
    Trained with PPO self-play + distillation from 4 sub-agents.
    At inference, uses OpponentStyleTracker to feed live opp embedding.
    """

    def __init__(self, obs_size: int, action_size: int,
                 lr: float = 2e-4, device: str = "cpu"):
        self.device      = torch.device(device)
        self.obs_size    = obs_size
        self.action_size = action_size
        self.opp_tracker = OpponentStyleTracker()

        self.network   = UnifiedNetwork(obs_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=lr, eps=1e-5)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    def _opp_tensor(self, batch_size: int = 1,
                    opp_emb: np.ndarray = None) -> torch.Tensor:
        if opp_emb is None:
            opp_emb = self.opp_tracker.embedding()
        t = torch.FloatTensor(opp_emb).unsqueeze(0).to(self.device)
        return t.expand(batch_size, -1)

    def observe_opponent(self, action: int, street: int,
                         faced_raise: bool = False):
        self.opp_tracker.update(action, street, faced_raise)

    def reset_opponent_tracker(self):
        self.opp_tracker.reset()

    def select_action(
        self, obs: np.ndarray, legal_actions: List[int],
        opp_emb: np.ndarray = None
    ) -> Tuple[int, float, float]:
        x         = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        opp_style = self._opp_tensor(1, opp_emb)
        mask      = torch.zeros(self.action_size)
        for a in legal_actions:
            if a < self.action_size:
                mask[a] = 1.0
        with torch.no_grad():
            dist, value = self.network.get_dist_and_value(
                x, opp_style, mask.to(self.device))
            action   = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def get_action_probs(
        self, obs: np.ndarray, legal_actions: List[int],
        opp_emb: np.ndarray = None
    ) -> np.ndarray:
        x         = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        opp_style = self._opp_tensor(1, opp_emb)
        mask      = torch.zeros(self.action_size)
        for a in legal_actions:
            if a < self.action_size:
                mask[a] = 1.0
        with torch.no_grad():
            logits, _ = self.network.forward(x, opp_style)
            logits    = logits.masked_fill(
                ~mask.bool().to(self.device), float('-inf'))
            probs = torch.softmax(logits, dim=-1)
        return probs.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # Distillation loss (called from train_unified.py)
    # ------------------------------------------------------------------
    def distill_loss(
        self,
        obs_t: torch.Tensor,
        opp_t: torch.Tensor,
        teacher_logits: torch.Tensor,  # [B, action_size] raw logits from teacher
        legal_mask: torch.Tensor,      # [B, action_size]
        temperature: float = 2.0,
    ) -> torch.Tensor:
        """
        KL divergence from teacher soft targets to student logits.
        teacher_logits can be the blended logits of multiple teachers.
        """
        student_logits, _ = self.network.forward(obs_t, opp_t)
        student_logits = student_logits.masked_fill(
            ~legal_mask.bool(), float('-inf'))

        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs     = F.softmax(teacher_logits    / temperature, dim=-1)
        loss = F.kl_div(student_log_probs, teacher_probs,
                        reduction='batchmean') * (temperature ** 2)
        return loss

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------
    def ppo_update(
        self,
        obs_list:     np.ndarray,
        opp_list:     np.ndarray,
        actions:      np.ndarray,
        old_logps:    np.ndarray,
        returns:      np.ndarray,
        values:       np.ndarray,
        legal_masks:  np.ndarray,
        teacher_logits_list: np.ndarray = None,  # [T, action_size] or None
        distill_coef: float = 0.0,
        clip_eps:     float = 0.2,
        entropy_coef: float = 0.05,
        temperature:  float = 2.0,
    ) -> dict:
        if len(obs_list) < 2:
            return {}

        obs_t    = torch.FloatTensor(obs_list).to(self.device)
        opp_t    = torch.FloatTensor(opp_list).to(self.device)
        act_t    = torch.LongTensor(actions).to(self.device)
        old_lp_t = torch.FloatTensor(old_logps).to(self.device)
        ret_t    = torch.FloatTensor(returns).to(self.device)
        adv_t    = ret_t - torch.FloatTensor(values).to(self.device)
        adv_t    = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        mask_t   = torch.FloatTensor(legal_masks).to(self.device)

        dist, new_values = self.network.get_dist_and_value(
            obs_t, opp_t, mask_t)
        new_logp = dist.log_prob(act_t)
        entropy  = dist.entropy().mean()

        ratio = (new_logp - old_lp_t).exp()
        surr  = torch.min(
            ratio * adv_t,
            ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_t
        )
        ppo_loss   = -surr.mean() - entropy_coef * entropy
        value_loss = F.mse_loss(new_values.squeeze(-1), ret_t)
        total_loss = ppo_loss + 0.5 * value_loss

        if distill_coef > 0 and teacher_logits_list is not None:
            tl_t  = torch.FloatTensor(teacher_logits_list).to(self.device)
            dloss = self.distill_loss(obs_t, opp_t, tl_t, mask_t, temperature)
            total_loss = total_loss + distill_coef * dloss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        return {
            "ppo":    ppo_loss.item(),
            "value":  value_loss.item(),
            "distill": distill_coef,
        }

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save(self.network.state_dict(), path)

    def load(self, path: str):
        self.network.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True))

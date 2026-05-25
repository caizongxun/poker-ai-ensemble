#!/usr/bin/env python
"""
train_unified.py

Three-phase training for UnifiedAgent:
  Phase 1 - Warm-up (warmup_steps):   pure distillation from 4 sub-agents
  Phase 2 - Main   (main_steps):      PPO self-play + distillation auxiliary
  Phase 3 - Anneal (anneal_steps):    PPO only, entropy & distill coef decay

Teacher blending:
  Each step, opponent is one of the 4 sub-agents (round-robin).
  Teacher logits = weighted blend of all 4 sub-agents:
    weight[i] proportional to counter_score[opp_style][i]
  counter_score encodes domain knowledge:
    vs aggressive  → deceptive(0.5) > conservative(0.3) > strategic(0.15) > aggressive(0.05)
    vs conservative→ aggressive(0.5) > deceptive(0.25) > strategic(0.2) > conservative(0.05)
    vs strategic   → deceptive(0.4) > aggressive(0.3) > conservative(0.2) > strategic(0.1)
    vs deceptive   → conservative(0.4) > strategic(0.3) > aggressive(0.2) > deceptive(0.1)
"""

import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.poker_env import PokerEnv
from agents.aggressive_agent   import AggressiveAgent
from agents.conservative_agent import ConservativeAgent
from agents.strategic_agent    import StrategicAgent
from agents.deceptive_agent    import DeceptiveAgent
from agents.unified_agent      import UnifiedAgent

SUB_AGENT_CLASSES = {
    "aggressive":   AggressiveAgent,
    "conservative": ConservativeAgent,
    "strategic":    StrategicAgent,
    "deceptive":    DeceptiveAgent,
}

# counter_weights[opp_name] = [aggressive_w, conservative_w, strategic_w, deceptive_w]
COUNTER_WEIGHTS = {
    "aggressive":   np.array([0.05, 0.30, 0.15, 0.50], dtype=np.float32),
    "conservative": np.array([0.50, 0.05, 0.20, 0.25], dtype=np.float32),
    "strategic":    np.array([0.30, 0.20, 0.10, 0.40], dtype=np.float32),
    "deceptive":    np.array([0.20, 0.40, 0.30, 0.10], dtype=np.float32),
}

# Style embedding per sub-agent (matches OpponentStyleTracker dims)
STYLE_EMBEDDINGS = {
    "aggressive":   np.array([0.80, 0.70, 0.85, 0.15], dtype=np.float32),
    "conservative": np.array([0.22, 0.12, 0.12, 0.75], dtype=np.float32),
    "strategic":    np.array([0.50, 0.40, 0.50, 0.40], dtype=np.float32),
    "deceptive":    np.array([0.55, 0.30, 0.60, 0.30], dtype=np.float32),
}


def load_sub_agents(checkpoint_dir: str, obs_size: int, action_size: int) -> dict:
    agents = {}
    for name, cls in SUB_AGENT_CLASSES.items():
        agent = cls(obs_size, action_size)
        path  = os.path.join(checkpoint_dir, f"{name}_final.pt")
        if os.path.exists(path):
            agent.load(path)
            print(f"  Loaded {name}")
        else:
            print(f"  WARNING: {path} not found, using random weights")
        agents[name] = agent
    return agents


def blended_teacher_logits(
    sub_agents: dict,
    obs: np.ndarray,
    legal_actions: list,
    opp_name: str,
    action_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Blend the 4 sub-agents' logits using counter_weights for opp_name.
    Returns raw blended logits [action_size] as numpy.
    """
    weights = COUNTER_WEIGHTS[opp_name]          # [4]
    names   = ["aggressive", "conservative", "strategic", "deceptive"]
    blended = np.zeros(action_size, dtype=np.float32)
    for i, name in enumerate(names):
        probs   = sub_agents[name].get_action_probs(obs, legal_actions)
        # convert probs back to logits (log to avoid -inf issues)
        logits  = np.log(probs + 1e-8)
        blended += weights[i] * logits
    return blended


def compute_returns(rewards: list, gamma: float = 0.99) -> np.ndarray:
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running     = rewards[t] + gamma * running
        returns[t]  = running
    return returns


def run_episode(
    env: PokerEnv,
    unified: UnifiedAgent,
    opp_agent,
    opp_name: str,
    sub_agents: dict,
    collect_teacher: bool,
) -> dict:
    """
    Run one episode. unified = player 0, opp_agent = player 1.
    Returns trajectory dict.
    """
    unified.reset_opponent_tracker()
    obs  = env.reset()
    done = False

    obs_list    = []
    opp_list    = []
    act_list    = []
    logp_list   = []
    val_list    = []
    rew_list    = []
    mask_list   = []
    teacher_list = [] if collect_teacher else None

    while not done:
        current = env.state.current_player
        legal   = env.get_legal_actions()

        if current == 0:
            opp_emb = unified.opp_tracker.embedding()
            action, logp, value = unified.select_action(obs, legal, opp_emb)

            # legal mask
            mask = np.zeros(unified.action_size, dtype=np.float32)
            for a in legal:
                if a < unified.action_size:
                    mask[a] = 1.0

            obs_list.append(obs.copy())
            opp_list.append(opp_emb.copy())
            act_list.append(action)
            logp_list.append(logp)
            val_list.append(value)
            rew_list.append(0.0)      # back-fill at end
            mask_list.append(mask)

            if collect_teacher:
                tl = blended_teacher_logits(
                    sub_agents, obs, legal, opp_name,
                    unified.action_size, unified.device)
                teacher_list.append(tl)

        else:
            opp_action, _, _ = opp_agent.select_action(obs, legal)
            street = getattr(env.state, "street", 0)
            unified.observe_opponent(
                opp_action, street, faced_raise=(opp_action >= 2))

        obs, _, done, _ = env.step(
            action if current == 0 else opp_action)

    if obs_list:
        rew_list[-1] = float(env.state.get_reward(0))

    return {
        "obs":     np.array(obs_list,     dtype=np.float32),
        "opp":     np.array(opp_list,     dtype=np.float32),
        "actions": np.array(act_list,     dtype=np.int64),
        "logps":   np.array(logp_list,    dtype=np.float32),
        "values":  np.array(val_list,     dtype=np.float32),
        "rewards": np.array(rew_list,     dtype=np.float32),
        "masks":   np.array(mask_list,    dtype=np.float32),
        "teacher": (np.array(teacher_list, dtype=np.float32)
                    if teacher_list else None),
    }


def train_unified(
    checkpoint_dir: str = "checkpoints",
    save_dir:       str = "checkpoints",
    warmup_steps:   int = 100_000,
    main_steps:     int = 500_000,
    anneal_steps:   int = 100_000,
    distill_coef_start: float = 0.30,
    distill_coef_end:   float = 0.05,
    entropy_coef_start: float = 0.05,
    entropy_coef_end:   float = 0.01,
    gamma:          float = 0.99,
):
    os.makedirs(save_dir, exist_ok=True)
    env        = PokerEnv(num_players=2)
    sub_agents = load_sub_agents(
        checkpoint_dir, env.observation_size, env.action_size)
    unified = UnifiedAgent(
        obs_size    = env.observation_size,
        action_size = env.action_size,
    )

    opp_names = list(sub_agents.keys())
    opp_agents = list(sub_agents.values())
    total_budget = warmup_steps + main_steps + anneal_steps

    print(f"Training UnifiedAgent for {total_budget:,} total steps")
    print(f"  Phase 1 warm-up:  {warmup_steps:,} steps (distillation only)")
    print(f"  Phase 2 main:     {main_steps:,} steps (PPO + distillation)")
    print(f"  Phase 3 anneal:   {anneal_steps:,} steps (PPO only, entropy decay)")

    pbar        = tqdm(total=total_budget)
    total_steps = 0
    opp_idx     = 0

    while total_steps < total_budget:
        # Determine phase
        if total_steps < warmup_steps:
            phase          = 1
            collect_teacher = True
            distill_coef   = distill_coef_start
            entropy_coef   = entropy_coef_start
        elif total_steps < warmup_steps + main_steps:
            phase           = 2
            collect_teacher = True
            t = (total_steps - warmup_steps) / main_steps
            distill_coef    = distill_coef_start + t * (
                distill_coef_end - distill_coef_start)
            entropy_coef    = entropy_coef_start
        else:
            phase           = 3
            collect_teacher = False
            distill_coef    = 0.0
            t = (total_steps - warmup_steps - main_steps) / anneal_steps
            entropy_coef    = entropy_coef_start + t * (
                entropy_coef_end - entropy_coef_start)

        opp_name  = opp_names[opp_idx % len(opp_names)]
        opp_agent = opp_agents[opp_idx % len(opp_agents)]
        opp_idx  += 1

        traj = run_episode(
            env, unified, opp_agent, opp_name,
            sub_agents, collect_teacher)

        n = len(traj["obs"])
        if n < 2:
            continue

        returns = compute_returns(traj["rewards"].tolist(), gamma)

        # Phase 1: distillation update only (no PPO)
        if phase == 1:
            if traj["teacher"] is not None and n > 0:
                obs_t    = torch.FloatTensor(traj["obs"]).to(unified.device)
                opp_t    = torch.FloatTensor(traj["opp"]).to(unified.device)
                tl_t     = torch.FloatTensor(traj["teacher"]).to(unified.device)
                mask_t   = torch.FloatTensor(traj["masks"]).to(unified.device)
                dloss    = unified.distill_loss(obs_t, opp_t, tl_t, mask_t, temperature=2.0)
                unified.optimizer.zero_grad()
                dloss.backward()
                torch.nn.utils.clip_grad_norm_(
                    unified.network.parameters(), 0.5)
                unified.optimizer.step()
        else:
            unified.ppo_update(
                obs_list    = traj["obs"],
                opp_list    = traj["opp"],
                actions     = traj["actions"],
                old_logps   = traj["logps"],
                returns     = returns,
                values      = traj["values"],
                legal_masks = traj["masks"],
                teacher_logits_list = traj["teacher"],
                distill_coef = distill_coef,
                entropy_coef = entropy_coef,
                temperature  = 2.0,
            )

        total_steps += n
        pbar.update(n)

        if total_steps % 50_000 < n:   # roughly every 50k steps
            ckpt = os.path.join(save_dir, f"unified_{total_steps}.pt")
            unified.save(ckpt)

    pbar.close()
    unified.save(os.path.join(save_dir, "unified_final.pt"))
    print(f"UnifiedAgent saved to {save_dir}/unified_final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--save-dir",       default="checkpoints")
    parser.add_argument("--warmup-steps",   type=int, default=100_000)
    parser.add_argument("--main-steps",     type=int, default=500_000)
    parser.add_argument("--anneal-steps",   type=int, default=100_000)
    args = parser.parse_args()
    train_unified(
        checkpoint_dir = args.checkpoint_dir,
        save_dir       = args.save_dir,
        warmup_steps   = args.warmup_steps,
        main_steps     = args.main_steps,
        anneal_steps   = args.anneal_steps,
    )

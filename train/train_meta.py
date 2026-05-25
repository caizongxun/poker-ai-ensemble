import argparse
import os
import sys
import pickle
import random
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.poker_env import PokerEnv
from agents.aggressive_agent import AggressiveAgent
from agents.conservative_agent import ConservativeAgent
from agents.strategic_agent import StrategicAgent
from agents.deceptive_agent import DeceptiveAgent
from meta.meta_agent import MetaAgent

SUB_AGENT_CLASSES = {
    "aggressive":   AggressiveAgent,
    "conservative": ConservativeAgent,
    "strategic":    StrategicAgent,
    "deceptive":    DeceptiveAgent,
}

_OPP_VPIP = 122
_OPP_PFR  = 123
_OPP_AF   = 124

# 手工風格 embedding (vpip, pfr, af_norm, fold_to_raise)
STYLE_EMBEDDINGS = {
    "aggressive":   np.array([0.80, 0.70, 0.85, 0.15], dtype=np.float32),
    "conservative": np.array([0.22, 0.12, 0.12, 0.75], dtype=np.float32),
    "strategic":    np.array([0.50, 0.40, 0.50, 0.40], dtype=np.float32),
    "deceptive":    np.array([0.55, 0.30, 0.60, 0.30], dtype=np.float32),
}
STYLE_TO_IDX = {"aggressive": 0, "conservative": 1,
                "strategic": 2, "deceptive": 3}


def load_sub_agents(checkpoint_dir, obs_size, action_size):
    agents = {}
    for name, cls in SUB_AGENT_CLASSES.items():
        agent = cls(obs_size, action_size)
        path  = os.path.join(checkpoint_dir, f"{name}_final.pt")
        if os.path.exists(path):
            agent.load(path)
            print(f"  Loaded {name}")
        else:
            print(f"  WARNING: {path} not found")
        agents[name] = agent
    return agents


def pretrain_gating(meta, battle_data_path, epochs=10):
    """
    修正版 pretrain：
    1. 只用 obs_a（player 0 的觀測）訓練，不混入 player 1 的 obs
    2. label = agent_a 的 expert idx（訓練 gating 展現「我自己的風格」）
       邏輯：當 agent_a 贏時訓練 gating 輸出 agent_a 的 expert weight 高。
              當 agent_a 輸時，訓練 gating 輸出 agent_b 的 expert weight 高（應該改用 b 的策略）。
    3. opp_style 用游所 obs_a 本身的對手統計向量提取 + 手工 embedding 混合
    """
    with open(battle_data_path, "rb") as f:
        records = pickle.load(f)
    print(f"Pretraining from {len(records)} battle records...")

    for epoch in range(epochs):
        random.shuffle(records)
        total_loss = 0.0
        count      = 0

        for rec in records:
            obs_list = rec.get("obs_a", [])   # 只用 player 0 的 obs
            if not obs_list:
                continue

            rew_a   = rec["rewards_a"][-1] if rec["rewards_a"] else 0.0
            agent_a = rec["agent_a"]
            agent_b = rec["agent_b"]

            # label 邏輯:
            # agent_a 贏 -> 展現 agent_a 的風格是對的 -> label = agent_a idx
            # agent_a 輸 -> 應該改用 agent_b 的風格 -> label = agent_b idx
            best_expert = (STYLE_TO_IDX.get(agent_a, 0) if rew_a > 0
                           else STYLE_TO_IDX.get(agent_b, 0))

            obs_t  = torch.FloatTensor(obs_list)  # [T, obs_size]
            T      = len(obs_list)
            target = torch.zeros(T, dtype=torch.long).fill_(best_expert)

            # opp_style: 前期用手工 embedding，後期用 obs 內對手統計
            style_emb    = torch.FloatTensor(
                STYLE_EMBEDDINGS.get(agent_b,
                    np.array([0.5]*4, dtype=np.float32))
            ).unsqueeze(0).expand(T, -1)                      # [T,4]
            opp_from_obs = obs_t[:, _OPP_VPIP:_OPP_VPIP+4].clamp(0, 1)  # [T,4]
            alphas       = (torch.arange(T, dtype=torch.float32) / T
                            ).unsqueeze(1)                     # [T,1]
            opp_style    = (1 - alphas) * style_emb + alphas * opp_from_obs

            weights = meta.gating(obs_t, opp_style)
            loss    = torch.nn.functional.cross_entropy(weights, target)
            entropy = -(weights * (weights + 1e-8).log()).sum(-1).mean()
            total   = loss - 0.1 * entropy

            meta.optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(meta.gating.parameters(), 0.5)
            meta.optimizer.step()
            total_loss += loss.item()
            count      += 1

        print(f"  Pretrain epoch {epoch+1}/{epochs}, "
              f"loss: {total_loss/max(count,1):.4f}")


def ppo_meta_update(meta, sub_agents, expert_names,
                    obs_list, opp_styles, actions, old_logps,
                    rewards, values, gamma=0.99, clip_eps=0.2):
    n = len(rewards)
    if n < 2:
        return

    returns = np.zeros(n, dtype=np.float32)
    running = 0.0
    for t in reversed(range(n)):
        running    = rewards[t] + gamma * running
        returns[t] = running

    obs_t      = torch.FloatTensor(np.array(obs_list,   dtype=np.float32))
    opp_t      = torch.FloatTensor(np.array(opp_styles, dtype=np.float32))
    act_t      = torch.LongTensor(actions)
    old_logp_t = torch.FloatTensor(old_logps)
    ret_t      = torch.FloatTensor(returns)
    adv_t      = ret_t - torch.FloatTensor(values)
    adv_t      = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    weights        = meta.gating(obs_t, opp_t)
    gating_entropy = -(weights * (weights + 1e-8).log()).sum(-1).mean()

    all_expert_probs = []
    for name in expert_names:
        probs_list = [
            sub_agents[name].get_action_probs(ob, list(range(meta.action_size)))
            for ob in obs_list
        ]
        all_expert_probs.append(torch.FloatTensor(np.array(probs_list)))
    expert_stack = torch.stack(all_expert_probs, dim=1)   # [N,4,8]
    mixed_probs  = (weights.unsqueeze(2) * expert_stack).sum(1)  # [N,8]
    mixed_probs  = mixed_probs.clamp(min=1e-8)
    mixed_probs  = mixed_probs / mixed_probs.sum(-1, keepdim=True)

    dist     = torch.distributions.Categorical(probs=mixed_probs)
    new_logp = dist.log_prob(act_t)
    ratio    = (new_logp - old_logp_t).exp()
    surr     = torch.min(ratio * adv_t,
                         ratio.clamp(1-clip_eps, 1+clip_eps) * adv_t)
    loss     = -surr.mean() - 0.05 * gating_entropy

    meta.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(meta.gating.parameters(), 0.5)
    meta.optimizer.step()


def train_meta(checkpoint_dir, battle_data_path, episodes, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    env        = PokerEnv(num_players=2)
    sub_agents = load_sub_agents(
        checkpoint_dir, env.observation_size, env.action_size)
    meta = MetaAgent(
        obs_size    = env.observation_size,
        action_size = env.action_size,
        sub_agents  = sub_agents,
        mode        = "soft",
    )
    expert_names = MetaAgent.EXPERT_NAMES
    opp_list     = list(sub_agents.values())
    opp_names    = list(sub_agents.keys())

    if os.path.exists(battle_data_path):
        print("Pre-training gating network from battle logs...")
        pretrain_gating(meta, battle_data_path, epochs=10)
    else:
        print(f"WARNING: {battle_data_path} not found, skipping pretrain")

    print(f"Fine-tuning Meta Agent for {episodes} episodes "
          f"(vs all sub-agents, round-robin)...")
    total_steps = 0
    opp_idx     = 0
    pbar        = tqdm(total=episodes)

    while total_steps < episodes:
        opp_agent = opp_list[opp_idx % len(opp_list)]
        opp_name  = opp_names[opp_idx % len(opp_names)]
        opp_idx  += 1

        meta.reset_opponent_tracker()
        obs  = env.reset()
        done = False
        ep_obs, ep_opp, ep_acts, ep_rews, ep_logps, ep_vals = \
            [], [], [], [], [], []

        while not done:
            current = env.state.current_player
            legal   = env.get_legal_actions()

            if current == 0:
                action, logp, value, _ = meta.select_action(obs, legal)
                ep_obs.append(obs)
                ep_opp.append(meta.opp_tracker.get_embedding())
                ep_acts.append(action)
                ep_logps.append(logp)
                ep_vals.append(value)
                ep_rews.append(
                    float(env.state.get_reward(0)) if done else 0.0)
            else:
                opp_action, _, _ = opp_agent.select_action(obs, legal)
                street      = getattr(env.state, "street", 0)
                meta.observe_opponent(opp_action, street,
                                      faced_raise=(opp_action >= 2))
            next_obs, _, done, _ = env.step(
                action if current == 0 else opp_action)
            if done and ep_obs:
                ep_rews[-1] = float(env.state.get_reward(0))
            obs          = next_obs
            total_steps += 1

        if len(ep_obs) > 1:
            ppo_meta_update(meta, sub_agents, expert_names,
                            ep_obs, ep_opp, ep_acts,
                            ep_logps, ep_rews, ep_vals)

        pbar.update(max(0, total_steps - pbar.n))
        if total_steps % 20000 == 0:
            meta.save(os.path.join(save_dir, f"meta_{total_steps}.pt"))

    pbar.close()
    meta.save(os.path.join(save_dir, "meta_final.pt"))
    print(f"Meta Agent saved to {save_dir}/meta_final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--battle-data",    default="data/battle_logs.pkl")
    parser.add_argument("--episodes",       type=int, default=500000)
    parser.add_argument("--save-dir",       default="checkpoints")
    args = parser.parse_args()
    train_meta(args.checkpoint_dir, args.battle_data,
               args.episodes, args.save_dir)

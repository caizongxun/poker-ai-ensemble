import argparse
import os
import sys
import pickle
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


def load_sub_agents(checkpoint_dir, obs_size, action_size):
    agents = {
        "aggressive": AggressiveAgent(obs_size, action_size),
        "conservative": ConservativeAgent(obs_size, action_size),
        "strategic": StrategicAgent(obs_size, action_size),
        "deceptive": DeceptiveAgent(obs_size, action_size),
    }
    for name, agent in agents.items():
        path = os.path.join(checkpoint_dir, f"{name}_final.pt")
        if os.path.exists(path):
            agent.load(path)
    return agents


def pretrain_gating(meta, battle_data_path, epochs=5):
    """從 battle logs 預訓練 gating network。"""
    with open(battle_data_path, "rb") as f:
        records = pickle.load(f)
    agent_idx = {"aggressive": 0, "conservative": 1, "strategic": 2, "deceptive": 3}
    for epoch in range(epochs):
        total_loss = 0
        count = 0
        for rec in records:
            if not rec["obs"]:
                continue
            obs_t = torch.FloatTensor(rec["obs"])
            rews = np.array(rec["rewards"])
            best_expert = agent_idx.get(
                rec.get("agent_a" if rews.mean() >= 0 else "agent_b", "aggressive"), 0
            )
            target = torch.zeros(len(obs_t), dtype=torch.long).fill_(best_expert)
            weights = meta.gating(obs_t)
            loss = torch.nn.functional.cross_entropy(weights, target)
            meta.optimizer.zero_grad()
            loss.backward()
            meta.optimizer.step()
            total_loss += loss.item()
            count += 1
        print(f"  Pretrain epoch {epoch+1}/{epochs}, loss: {total_loss/max(count,1):.4f}")


def ppo_meta_update(meta, obs, actions, old_logps, rewards, values,
                    gamma=0.99, clip_eps=0.2):
    n = len(rewards)
    if n < 2:
        return
    returns = np.zeros(n, dtype=np.float32)
    running = 0.0
    for t in reversed(range(n)):
        running = rewards[t] + gamma * running
        returns[t] = running
    returns_t = torch.FloatTensor(returns)
    obs_t = torch.FloatTensor(np.array(obs, dtype=np.float32))
    act_t = torch.LongTensor(actions)
    old_logp_t = torch.FloatTensor(old_logps)
    adv_t = returns_t - torch.FloatTensor(values)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    weights = meta.gating(obs_t)
    dist = torch.distributions.Categorical(probs=weights)
    new_logp = dist.log_prob(act_t)
    ratio = (new_logp - old_logp_t).exp()
    surr = torch.min(ratio * adv_t, ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_t)
    loss = -surr.mean()
    meta.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(meta.gating.parameters(), 0.5)
    meta.optimizer.step()


def train_meta(checkpoint_dir, battle_data_path, episodes, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    env = PokerEnv(num_players=2)
    sub_agents = load_sub_agents(checkpoint_dir, env.observation_size, env.action_size)
    meta = MetaAgent(
        obs_size=env.observation_size,
        action_size=env.action_size,
        sub_agents=sub_agents,
        mode="soft",
    )
    if os.path.exists(battle_data_path):
        print("Pre-training gating network from battle logs...")
        pretrain_gating(meta, battle_data_path, epochs=5)
    print(f"Fine-tuning Meta Agent for {episodes} episodes...")
    total_steps = 0
    pbar = tqdm(total=episodes)
    while total_steps < episodes:
        obs = env.reset()
        ep_obs, ep_acts, ep_rews, ep_logps, ep_vals = [], [], [], [], []
        done = False
        while not done:
            legal = env.get_legal_actions()
            action, logp, value, weights = meta.select_action(obs, legal)
            next_obs, reward, done, _ = env.step(action)
            ep_obs.append(obs)
            ep_acts.append(action)
            ep_rews.append(reward)
            ep_logps.append(logp)
            ep_vals.append(value)
            obs = next_obs
            total_steps += 1
        if len(ep_obs) > 1:
            ppo_meta_update(meta, ep_obs, ep_acts, ep_logps, ep_rews, ep_vals)
        pbar.update(len(ep_obs))
        if total_steps % 20000 == 0:
            meta.save(os.path.join(save_dir, f"meta_{total_steps}.pt"))
    pbar.close()
    meta.save(os.path.join(save_dir, "meta_final.pt"))
    print(f"Meta Agent saved to {save_dir}/meta_final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--battle-data", default="data/battle_logs.pkl")
    parser.add_argument("--episodes", type=int, default=200000)
    parser.add_argument("--save-dir", default="checkpoints")
    args = parser.parse_args()
    train_meta(args.checkpoint_dir, args.battle_data, args.episodes, args.save_dir)

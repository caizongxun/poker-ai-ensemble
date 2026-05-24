import argparse
import os
import random
import sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.poker_env import PokerEnv
from agents.aggressive_agent import AggressiveAgent
from agents.conservative_agent import ConservativeAgent
from agents.strategic_agent import StrategicAgent
from agents.deceptive_agent import DeceptiveAgent

AGENT_MAP = {
    "aggressive": AggressiveAgent,
    "conservative": ConservativeAgent,
    "strategic": StrategicAgent,
    "deceptive": DeceptiveAgent,
}


def collect_selfplay_rollout(env, agent, num_steps=2048):
    """
    Self-play rollout: agent 同時扑演 player 0 和 player 1。
    每個 step 根據 env.state.current_player 路由。
    只收集 agent 主動行動的那一步，不收集對手步驟。
    """
    obs_buf, act_buf, rew_buf, logp_buf, val_buf, done_buf = [], [], [], [], [], []
    obs = env.reset()
    # 随機指定此局 agent 扇演哪個位置 (0 或 1)，讓訓練更均勻
    agent_seat = random.randint(0, 1)
    step = 0
    while step < num_steps:
        current = env.state.current_player
        legal = env.get_legal_actions()
        action, logp, value = agent.select_action(obs, legal)
        next_obs, reward, done, _ = env.step(action)
        if current == agent_seat:
            # 只收集 agent 自己那一側的資料
            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(logp)
            val_buf.append(value)
            if done:
                final_reward = env.state.get_reward(agent_seat)
                shaped = agent.compute_reward_shaping(action, obs, final_reward)
                rew_buf.append(shaped)
                done_buf.append(True)
            else:
                rew_buf.append(0.0)
                done_buf.append(False)
            step += 1
        if done:
            obs = env.reset()
            agent_seat = random.randint(0, 1)
        else:
            obs = next_obs
    return (
        np.array(obs_buf, dtype=np.float32),
        np.array(act_buf),
        np.array(rew_buf, dtype=np.float32),
        np.array(logp_buf, dtype=np.float32),
        np.array(val_buf, dtype=np.float32),
        np.array(done_buf),
    )


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_val = 0.0 if dones[t] else (values[t + 1] if t + 1 < n else 0.0)
        delta = rewards[t] + gamma * next_val - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (0 if dones[t] else last_gae)
    returns = advantages + values
    return advantages, returns


def ppo_update(agent, obs, actions, old_logps, advantages, returns,
               clip_eps=0.2, epochs=4, batch_size=256):
    n = len(obs)
    obs_t = torch.FloatTensor(obs)
    act_t = torch.LongTensor(actions)
    old_logp_t = torch.FloatTensor(old_logps)
    adv_t = torch.FloatTensor(advantages)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    ret_t = torch.FloatTensor(returns)
    for _ in range(epochs):
        idx = torch.randperm(n)
        for start in range(0, n, batch_size):
            batch = idx[start: start + batch_size]
            dist, values = agent.network.get_action_distribution(obs_t[batch])
            new_logp = dist.log_prob(act_t[batch])
            ratio = (new_logp - old_logp_t[batch]).exp()
            surr1 = ratio * adv_t[batch]
            surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_t[batch]
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (values.squeeze() - ret_t[batch]).pow(2).mean()
            entropy = dist.entropy().mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.network.parameters(), 0.5)
            agent.optimizer.step()


def train(agent_name: str, episodes: int, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    env = PokerEnv(num_players=2)
    AgentClass = AGENT_MAP[agent_name]
    agent = AgentClass(env.observation_size, env.action_size)
    print(f"Training {agent_name} agent for {episodes} steps (self-play)...")
    total_steps = 0
    pbar = tqdm(total=episodes)
    while total_steps < episodes:
        obs, acts, rews, logps, vals, dones = collect_selfplay_rollout(env, agent)
        advs, rets = compute_gae(rews, vals, dones)
        ppo_update(agent, obs, acts, logps, advs, rets)
        total_steps += len(obs)
        pbar.update(len(obs))
        if total_steps % 50000 == 0:
            agent.save(os.path.join(save_dir, f"{agent_name}_{total_steps}.pt"))
    pbar.close()
    agent.save(os.path.join(save_dir, f"{agent_name}_final.pt"))
    print(f"Saved: {save_dir}/{agent_name}_final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=list(AGENT_MAP.keys()), required=True)
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--save-dir", default="checkpoints")
    args = parser.parse_args()
    train(args.agent, args.episodes, args.save_dir)

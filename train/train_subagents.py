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
    "aggressive":   AggressiveAgent,
    "conservative": ConservativeAgent,
    "strategic":    StrategicAgent,
    "deceptive":    DeceptiveAgent,
}


def load_opponent(name, obs_size, action_size, checkpoint_dir):
    if name not in AGENT_MAP:
        return None
    opp  = AGENT_MAP[name](obs_size, action_size)
    path = os.path.join(checkpoint_dir, f"{name}_final.pt")
    if os.path.exists(path):
        opp.load(path)
        print(f"  Opponent loaded: {name}")
    else:
        print(f"  Opponent {name} checkpoint not found, using random weights")
    return opp


def collect_rollout(env, agent, opponent, num_steps=2048, agent_seat=None):
    """
    修正：將 step_action 與 current 綁定在同一個 if/else 內部。
    不再後期用 (action if current==0 else opp_action) 的对據式。
    """
    obs_buf, act_buf, rew_buf, logp_buf, val_buf, done_buf = \
        [], [], [], [], [], []
    obs  = env.reset()
    seat = random.randint(0, 1) if agent_seat is None else agent_seat
    step = 0

    while step < num_steps:
        current = env.state.current_player
        legal   = env.get_legal_actions()

        if current == seat:
            action, logp, value = agent.select_action(obs, legal)
            next_obs, _, done, _ = env.step(action)
            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(logp)
            val_buf.append(value)
            if done:
                final_reward = env.state.get_reward(seat)
                shaped = agent.compute_reward_shaping(action, obs, final_reward)
                rew_buf.append(shaped)
                done_buf.append(True)
            else:
                rew_buf.append(0.0)
                done_buf.append(False)
            step += 1
        else:
            # 對手座位：用 opponent 或 self-play
            if opponent is not None:
                opp_action, _, _ = opponent.select_action(obs, legal)
            else:
                opp_action, _, _ = agent.select_action(obs, legal)
            next_obs, _, done, _ = env.step(opp_action)

        if done:
            obs  = env.reset()
            seat = random.randint(0, 1) if agent_seat is None else agent_seat
        else:
            obs = next_obs

    return (
        np.array(obs_buf,  dtype=np.float32),
        np.array(act_buf),
        np.array(rew_buf,  dtype=np.float32),
        np.array(logp_buf, dtype=np.float32),
        np.array(val_buf,  dtype=np.float32),
        np.array(done_buf),
    )


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    n          = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae   = 0.0
    for t in reversed(range(n)):
        next_val      = 0.0 if dones[t] else (values[t + 1] if t + 1 < n else 0.0)
        delta         = rewards[t] + gamma * next_val - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (0 if dones[t] else last_gae)
    return advantages, advantages + values


def ppo_update(agent, obs, actions, old_logps, advantages, returns,
               clip_eps=0.2, epochs=4, batch_size=256):
    n          = len(obs)
    obs_t      = torch.FloatTensor(obs)
    act_t      = torch.LongTensor(actions)
    old_logp_t = torch.FloatTensor(old_logps)
    adv_t      = torch.FloatTensor(advantages)
    adv_t      = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    ret_t      = torch.FloatTensor(returns)

    for _ in range(epochs):
        idx = torch.randperm(n)
        for start in range(0, n, batch_size):
            batch    = idx[start: start + batch_size]
            dist, values = agent.network.get_action_distribution(obs_t[batch])
            new_logp = dist.log_prob(act_t[batch])
            ratio    = (new_logp - old_logp_t[batch]).exp()
            surr1    = ratio * adv_t[batch]
            surr2    = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_t[batch]
            loss     = (-torch.min(surr1, surr2).mean()
                        + 0.5 * (values.squeeze() - ret_t[batch]).pow(2).mean()
                        - 0.01 * dist.entropy().mean())
            agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.network.parameters(), 0.5)
            agent.optimizer.step()


def train(agent_name, episodes, save_dir, checkpoint_dir,
          opponent_names=None, opponent_ratio=0.5):
    os.makedirs(save_dir, exist_ok=True)
    env        = PokerEnv(num_players=2)
    AgentClass = AGENT_MAP[agent_name]
    agent      = AgentClass(env.observation_size, env.action_size)

    opponents = []
    if opponent_names:
        for oname in opponent_names:
            opp = load_opponent(oname, env.observation_size,
                                env.action_size, checkpoint_dir)
            if opp:
                opponents.append(opp)

    mode_str = (f"vs {opponent_names}" if opponents else "self-play")
    print(f"Training {agent_name} [{mode_str}] for {episodes} steps...")

    total_steps = 0
    opp_idx     = 0
    pbar        = tqdm(total=episodes)

    while total_steps < episodes:
        if opponents and random.random() < opponent_ratio:
            opp = opponents[opp_idx % len(opponents)]
            opp_idx += 1
        else:
            opp = None

        obs, acts, rews, logps, vals, dones = \
            collect_rollout(env, agent, opp)
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
    parser.add_argument("--agent",    choices=list(AGENT_MAP.keys()), required=True)
    parser.add_argument("--episodes", type=int, default=300000)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument(
        "--opponents", nargs="*", default=None,
        help="對戰的對手名稱，可多個。不指定則純 self-play。"
    )
    parser.add_argument(
        "--opponent-ratio", type=float, default=0.6,
        help="對戰局占總局數比例 (0~1)。"
    )
    args = parser.parse_args()
    train(args.agent, args.episodes, args.save_dir,
          args.checkpoint_dir, args.opponents, args.opponent_ratio)

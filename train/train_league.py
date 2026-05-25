import argparse
import os
import sys
import random
import numpy as np
import torch
from itertools import combinations
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


def load_agents(checkpoint_dir, obs_size, action_size):
    agents = {}
    for name, cls in AGENT_MAP.items():
        agent = cls(obs_size, action_size)
        path = os.path.join(checkpoint_dir, f"{name}_final.pt")
        if os.path.exists(path):
            agent.load(path)
            print(f"  Loaded {name} from {path}")
        else:
            print(f"  Initialized {name} (no checkpoint found)")
        agents[name] = agent
    return agents


def collect_crossplay_rollout(env, agent_a, agent_b, num_steps=512):
    """
    agent_a = player 0, agent_b = player 1.
    兩邊同時收集 rollout。

    修正重點：
    - 局結束時兩邊同時將真實 reward 寫入各自最後一個 step
    - 中途未結束的 step reward = 0，done = False，不影響 GAE
    - 兩邊都用各自的 obs（對应各自的觀察角度）
    """
    buf = {
        0: dict(obs=[], act=[], rew=[], logp=[], val=[], done=[]),
        1: dict(obs=[], act=[], rew=[], logp=[], val=[], done=[]),
    }
    agents = {0: agent_a, 1: agent_b}
    obs = env.reset()
    step = 0

    while step < num_steps:
        current = env.state.current_player
        legal = env.get_legal_actions()
        agent = agents[current]

        action, logp, value = agent.select_action(obs, legal)
        next_obs, _, done, _ = env.step(action)

        buf[current]["obs"].append(obs.copy())
        buf[current]["act"].append(action)
        buf[current]["logp"].append(logp)
        buf[current]["val"].append(value)

        if done:
            # 局結束：兩邊同時寫入真實 shaped reward
            for seat in (0, 1):
                base_r = env.state.get_reward(seat)
                last_obs = buf[seat]["obs"][-1] if buf[seat]["obs"] else obs
                last_act = buf[seat]["act"][-1] if buf[seat]["act"] else action
                shaped = agents[seat].compute_reward_shaping(last_act, last_obs, base_r)
                if buf[seat]["rew"]:
                    # 將 shaped reward 加到最後一個 step
                    buf[seat]["rew"][-1] = shaped
                    buf[seat]["done"][-1] = True
                else:
                    # 万一某座這局都沒有行動機會（極少發生）
                    buf[seat]["rew"].append(shaped)
                    buf[seat]["done"].append(True)
                    buf[seat]["logp"].append(0.0)
                    buf[seat]["val"].append(0.0)
            obs = env.reset()
        else:
            # 局未結束：只給行動方加入 step，reward=0, done=False
            buf[current]["rew"].append(0.0)
            buf[current]["done"].append(False)
            obs = next_obs

        step += 1

    result = {}
    for seat in (0, 1):
        d = buf[seat]
        n = min(len(d["obs"]), len(d["act"]), len(d["rew"]),
                len(d["logp"]), len(d["val"]), len(d["done"]))
        if n < 2:
            result[seat] = None
            continue
        result[seat] = (
            np.array(d["obs"][:n], dtype=np.float32),
            np.array(d["act"][:n], dtype=np.int64),
            np.array(d["rew"][:n], dtype=np.float32),
            np.array(d["logp"][:n], dtype=np.float32),
            np.array(d["val"][:n], dtype=np.float32),
            np.array(d["done"][:n], dtype=bool),
        )
    return result


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_val = 0.0 if dones[t] else (values[t + 1] if t + 1 < n else 0.0)
        delta = rewards[t] + gamma * next_val - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (0.0 if dones[t] else last_gae)
    returns = advantages + values
    return advantages, returns


def ppo_update(agent, obs, actions, old_logps, advantages, returns,
               clip_eps=0.2, epochs=4, batch_size=256):
    n = len(obs)
    if n < 2:
        return
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


def train_league(checkpoint_dir, total_steps, save_dir, rollout_steps=512,
                 save_interval=100000, log_interval=20000):
    os.makedirs(save_dir, exist_ok=True)
    env = PokerEnv(num_players=2)
    agents = load_agents(checkpoint_dir, env.observation_size, env.action_size)
    names = list(agents.keys())
    pairs = list(combinations(names, 2))  # 6 pairs

    step_counts  = {n: 0 for n in names}
    # WinRate: 用局結束時的 base_reward > 0 計算，不用 shaped reward
    win_counts   = {n: 0 for n in names}
    hand_counts  = {n: 0 for n in names}

    print(f"\nLeague training for {total_steps:,} total steps...")
    print(f"Pairs: {pairs}\n")
    pbar = tqdm(total=total_steps)
    global_steps = 0

    while global_steps < total_steps:
        a_name, b_name = random.choice(pairs)
        if random.random() < 0.5:
            a_name, b_name = b_name, a_name

        rollout = collect_crossplay_rollout(
            env, agents[a_name], agents[b_name], num_steps=rollout_steps
        )

        name_map = {0: a_name, 1: b_name}

        for seat in (0, 1):
            name = name_map[seat]
            data = rollout[seat]
            if data is None:
                continue
            obs, acts, rews, logps, vals, dones = data
            advs, rets = compute_gae(rews, vals, dones)
            ppo_update(agents[name], obs, acts, logps, advs, rets)
            step_counts[name] += len(obs)
            global_steps += len(obs)

            # 用 done 標記的局結束 reward 來計算真實勝率
            for i, d in enumerate(dones):
                if d:
                    hand_counts[name] += 1
                    if rews[i] > 0:
                        win_counts[name] += 1

        pbar.update(rollout_steps * 2)

        if global_steps % log_interval < rollout_steps * 2:
            print(f"\n[{global_steps:,} steps]")
            for name in names:
                wr = win_counts[name] / max(hand_counts[name], 1)
                print(f"  {name:15s}: {step_counts[name]:7,} steps  "
                      f"WinRate={wr:5.1%}  "
                      f"({win_counts[name]}/{hand_counts[name]} hands)")

        if global_steps % save_interval < rollout_steps * 2:
            for name, agent in agents.items():
                agent.save(os.path.join(save_dir, f"{name}_league_{global_steps}.pt"))
            print(f"  [ckpt] saved at {global_steps:,} steps")

    pbar.close()

    for name, agent in agents.items():
        path = os.path.join(save_dir, f"{name}_final.pt")
        agent.save(path)
        print(f"  Saved final: {path}")

    print("\nFinal WinRates:")
    for name in names:
        wr = win_counts[name] / max(hand_counts[name], 1)
        print(f"  {name:15s}: WinRate={wr:.1%}  ({win_counts[name]}/{hand_counts[name]} hands)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--total-steps", type=int, default=2000000)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--save-interval", type=int, default=100000)
    parser.add_argument("--log-interval", type=int, default=20000)
    args = parser.parse_args()
    train_league(
        args.checkpoint_dir,
        args.total_steps,
        args.save_dir,
        args.rollout_steps,
        args.save_interval,
        args.log_interval,
    )

import argparse
import os
import sys
import pickle
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.poker_env import PokerEnv
from agents.aggressive_agent import AggressiveAgent
from agents.conservative_agent import ConservativeAgent
from agents.strategic_agent import StrategicAgent
from agents.deceptive_agent import DeceptiveAgent


def load_agents(checkpoint_dir, obs_size, action_size):
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
            print(f"  Loaded {name}")
        else:
            print(f"  WARNING: {path} not found, using random weights")
    return agents


def run_battle(env, agent_a, agent_b, episodes=1000):
    """
    agent_a = player 0, agent_b = player 1。
    根據 env.state.current_player 路由行動。
    """
    records = []
    for _ in range(episodes):
        obs = env.reset()
        episode = {"obs": [], "actions": [], "rewards_a": [], "rewards_b": [],
                   "current_players": []}
        done = False
        last_player = 0
        while not done:
            current = env.state.current_player
            legal = env.get_legal_actions()
            if current == 0:
                action, _, _ = agent_a.select_action(obs, legal)
            else:
                action, _, _ = agent_b.select_action(obs, legal)
            last_player = current
            next_obs, reward, done, _ = env.step(action)
            episode["obs"].append(obs.tolist())
            episode["actions"].append(action)
            episode["current_players"].append(current)
            if done:
                episode["rewards_a"].append(float(env.state.get_reward(0)))
                episode["rewards_b"].append(float(env.state.get_reward(1)))
            else:
                episode["rewards_a"].append(0.0)
                episode["rewards_b"].append(0.0)
            obs = next_obs
        records.append(episode)
    return records


def generate(checkpoint_dir, output_path, episodes_per_pair):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    env = PokerEnv(num_players=2)
    agents = load_agents(checkpoint_dir, env.observation_size, env.action_size)
    names = list(agents.keys())
    all_records = []
    # 包含 self-play (a vs a) 和 cross-play (a vs b)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i:]]
    for a_name, b_name in tqdm(pairs, desc="Battle pairs"):
        records = run_battle(env, agents[a_name], agents[b_name], episodes_per_pair)
        for r in records:
            r["agent_a"] = a_name
            r["agent_b"] = b_name
        all_records.extend(records)
        # 简單統計輸出
        wins_a = sum(1 for r in records if r["rewards_a"][-1] > 0)
        print(f"  {a_name} vs {b_name}: {wins_a}/{len(records)} wins for {a_name}")
    with open(output_path, "wb") as f:
        pickle.dump(all_records, f)
    print(f"\nSaved {len(all_records)} battle records to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--output", default="data/battle_logs.pkl")
    parser.add_argument("--episodes-per-pair", type=int, default=5000)
    args = parser.parse_args()
    generate(args.checkpoint_dir, args.output, args.episodes_per_pair)

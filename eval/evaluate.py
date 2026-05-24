import os
import sys
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.poker_env import PokerEnv
from agents.aggressive_agent import AggressiveAgent
from agents.conservative_agent import ConservativeAgent
from agents.strategic_agent import StrategicAgent
from agents.deceptive_agent import DeceptiveAgent
from meta.meta_agent import MetaAgent

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
        agents[name] = agent
    return agents


def play_match(env, agent_a, agent_b, num_hands: int):
    """
    讓兩個 agent 對打 num_hands 局。
    回傳 (agent_a 總収益, agent_b 總収益)。
    """
    rewards_a, rewards_b = [], []
    for _ in range(num_hands):
        obs = env.reset()
        done = False
        total_a, total_b = 0.0, 0.0
        turn = 0
        while not done:
            current = agent_a if turn % 2 == 0 else agent_b
            legal = env.get_legal_actions()
            action, _, _ = current.select_action(obs, legal)
            obs, reward, done, _ = env.step(action)
            if done:
                if turn % 2 == 0:
                    total_a += reward
                    total_b -= reward
                else:
                    total_b += reward
                    total_a -= reward
            turn += 1
        rewards_a.append(total_a)
        rewards_b.append(total_b)
    return rewards_a, rewards_b


def play_match_meta(env, meta, opponent, num_hands: int):
    """Meta agent vs opponent，同時記錄 gating weights。"""
    rewards_meta = []
    all_weights = []
    for _ in range(num_hands):
        obs = env.reset()
        done = False
        total = 0.0
        turn = 0
        hand_weights = []
        while not done:
            legal = env.get_legal_actions()
            if turn % 2 == 0:  # meta 先行動
                action, _, _, weights = meta.select_action(obs, legal)
                hand_weights.append(weights)
            else:
                action, _, _ = opponent.select_action(obs, legal)
            obs, reward, done, _ = env.step(action)
            if done and turn % 2 == 0:
                total += reward
            turn += 1
        rewards_meta.append(total)
        if hand_weights:
            all_weights.append(np.mean(hand_weights, axis=0))
    return rewards_meta, all_weights


def bb_per_100(rewards, big_blind=2):
    """計算 BB/100 hands，撲克標準評估指標。"""
    return (np.mean(rewards) / big_blind) * 100


def winrate(rewards):
    return np.mean([1 if r > 0 else 0 for r in rewards])


def evaluate_subagents(checkpoint_dir, num_hands=1000):
    print("\n" + "=" * 60)
    print("SUB-AGENT HEAD-TO-HEAD EVALUATION")
    print("=" * 60)
    env = PokerEnv()
    agents = load_agents(checkpoint_dir, env.observation_size, env.action_size)
    names = list(agents.keys())

    results = defaultdict(lambda: {"wins": 0, "losses": 0, "bb100": []})

    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    for a_name, b_name in pairs:
        rews_a, rews_b = play_match(env, agents[a_name], agents[b_name], num_hands)
        bb_a = bb_per_100(rews_a)
        bb_b = bb_per_100(rews_b)
        wr_a = winrate(rews_a)
        print(f"\n{a_name:15s} vs {b_name:15s} ({num_hands} hands)")
        print(f"  {a_name:15s}: BB/100 = {bb_a:+.1f}  WinRate = {wr_a:.1%}")
        print(f"  {b_name:15s}: BB/100 = {bb_b:+.1f}  WinRate = {1-wr_a:.1%}")
        results[a_name]["bb100"].append(bb_a)
        results[b_name]["bb100"].append(bb_b)

    print("\n" + "-" * 60)
    print("OVERALL BB/100 (average across all opponents)")
    print("-" * 60)
    ranking = sorted(names, key=lambda n: np.mean(results[n]["bb100"]), reverse=True)
    for i, name in enumerate(ranking):
        avg = np.mean(results[name]["bb100"])
        print(f"  #{i+1} {name:20s}: {avg:+.1f} BB/100")


def evaluate_meta(checkpoint_dir, num_hands=1000):
    print("\n" + "=" * 60)
    print("META AGENT EVALUATION")
    print("=" * 60)
    env = PokerEnv()
    sub_agents = load_agents(checkpoint_dir, env.observation_size, env.action_size)
    meta = MetaAgent(
        obs_size=env.observation_size,
        action_size=env.action_size,
        sub_agents=sub_agents,
        mode="soft",
    )
    meta_path = os.path.join(checkpoint_dir, "meta_final.pt")
    if os.path.exists(meta_path):
        meta.load(meta_path)
    else:
        print("Warning: meta_final.pt not found, using random weights")

    expert_names = MetaAgent.EXPERT_NAMES
    for opp_name, opp_agent in sub_agents.items():
        rews, weights = play_match_meta(env, meta, opp_agent, num_hands)
        bb = bb_per_100(rews)
        wr = winrate(rews)
        avg_w = np.mean(weights, axis=0) if weights else np.ones(4) / 4
        print(f"\nMeta vs {opp_name:15s} ({num_hands} hands)")
        print(f"  BB/100 = {bb:+.1f}  WinRate = {wr:.1%}")
        print(f"  Avg Gating Weights:")
        for ename, w in zip(expert_names, avg_w):
            bar = '#' * int(w * 30)
            print(f"    {ename:15s}: {w:.3f}  |{bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--num-hands", type=int, default=1000,
                        help="每組對打局數，建議至少 1000")
    parser.add_argument("--mode", choices=["sub", "meta", "all"], default="all")
    args = parser.parse_args()

    if args.mode in ("sub", "all"):
        evaluate_subagents(args.checkpoint_dir, args.num_hands)
    if args.mode in ("meta", "all"):
        evaluate_meta(args.checkpoint_dir, args.num_hands)


if __name__ == "__main__":
    main()

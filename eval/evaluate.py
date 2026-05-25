import os
import sys
import argparse
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.poker_env import PokerEnv
from agents.aggressive_agent   import AggressiveAgent
from agents.conservative_agent import ConservativeAgent
from agents.strategic_agent    import StrategicAgent
from agents.deceptive_agent    import DeceptiveAgent
from agents.unified_agent      import UnifiedAgent

try:
    from meta.meta_agent import MetaAgent
    _HAS_META = True
except Exception:
    _HAS_META = False

AGENT_MAP = {
    "aggressive":   AggressiveAgent,
    "conservative": ConservativeAgent,
    "strategic":    StrategicAgent,
    "deceptive":    DeceptiveAgent,
}


def load_sub_agents(checkpoint_dir, obs_size, action_size):
    agents = {}
    for name, cls in AGENT_MAP.items():
        agent = cls(obs_size, action_size)
        path  = os.path.join(checkpoint_dir, f"{name}_final.pt")
        if os.path.exists(path):
            agent.load(path)
            print(f"  Loaded {name}")
        else:
            print(f"  WARNING: {path} not found")
        agents[name] = agent
    return agents


def play_match(env, agent_a, agent_b, num_hands: int):
    rewards_a, rewards_b = [], []
    for _ in range(num_hands):
        obs  = env.reset()
        done = False
        while not done:
            current = env.state.current_player
            legal   = env.get_legal_actions()
            if current == 0:
                action, _, _ = agent_a.select_action(obs, legal)
            else:
                action, _, _ = agent_b.select_action(obs, legal)
            obs, _, done, _ = env.step(action)
        rewards_a.append(env.state.get_reward(0))
        rewards_b.append(env.state.get_reward(1))
    return rewards_a, rewards_b


def play_match_unified(env, unified: UnifiedAgent, opponent, num_hands: int):
    """unified = player 0, opponent = player 1."""
    rewards = []
    for _ in range(num_hands):
        unified.reset_opponent_tracker()
        obs  = env.reset()
        done = False
        while not done:
            current = env.state.current_player
            legal   = env.get_legal_actions()
            if current == 0:
                action, _, _ = unified.select_action(obs, legal)
            else:
                opp_action, _, _ = opponent.select_action(obs, legal)
                street = getattr(env.state, "street", 0)
                unified.observe_opponent(
                    opp_action, street, faced_raise=(opp_action >= 2))
                action = opp_action
            obs, _, done, _ = env.step(action)
        rewards.append(env.state.get_reward(0))
    return rewards


def play_match_meta(env, meta, opponent, num_hands: int):
    rewards_meta = []
    all_weights  = []
    for _ in range(num_hands):
        obs  = env.reset()
        done = False
        hand_weights = []
        while not done:
            current = env.state.current_player
            legal   = env.get_legal_actions()
            if current == 0:
                action, _, _, weights = meta.select_action(obs, legal)
                hand_weights.append(weights)
            else:
                action, _, _ = opponent.select_action(obs, legal)
            obs, _, done, _ = env.step(action)
        rewards_meta.append(env.state.get_reward(0))
        if hand_weights:
            all_weights.append(np.mean(hand_weights, axis=0))
    return rewards_meta, all_weights


def bb_per_100(rewards, big_blind=2):
    return (np.mean(rewards) / big_blind) * 100


def winrate(rewards):
    return sum(1 for r in rewards if r > 0) / len(rewards)


def evaluate_subagents(checkpoint_dir, num_hands=1000):
    print("\n" + "=" * 60)
    print("SUB-AGENT HEAD-TO-HEAD EVALUATION")
    print("=" * 60)
    env    = PokerEnv()
    agents = load_sub_agents(
        checkpoint_dir, env.observation_size, env.action_size)
    names   = list(agents.keys())
    results = defaultdict(list)
    pairs   = [(a, b) for i, a in enumerate(names)
               for b in names[i + 1:]]
    for a_name, b_name in pairs:
        print(f"\nRunning: {a_name} vs {b_name} ({num_hands} hands)...",
              flush=True)
        rews_a, rews_b = play_match(
            env, agents[a_name], agents[b_name], num_hands)
        bb_a = bb_per_100(rews_a)
        bb_b = bb_per_100(rews_b)
        wr_a = winrate(rews_a)
        print(f"  {a_name:15s}: BB/100={bb_a:+6.1f}  WinRate={wr_a:.1%}")
        print(f"  {b_name:15s}: BB/100={bb_b:+6.1f}  WinRate={1 - wr_a:.1%}")
        results[a_name].append(bb_a)
        results[b_name].append(bb_b)
    print("\n" + "-" * 60)
    print("OVERALL BB/100")
    print("-" * 60)
    for i, name in enumerate(
            sorted(names, key=lambda n: np.mean(results[n]), reverse=True)):
        print(f"  #{i+1} {name:20s}: {np.mean(results[name]):+.1f} BB/100")


def evaluate_unified(checkpoint_dir, num_hands=1000):
    print("\n" + "=" * 60)
    print("UNIFIED AGENT EVALUATION")
    print("=" * 60)
    env        = PokerEnv()
    sub_agents = load_sub_agents(
        checkpoint_dir, env.observation_size, env.action_size)
    unified = UnifiedAgent(
        obs_size    = env.observation_size,
        action_size = env.action_size,
    )
    unified_path = os.path.join(checkpoint_dir, "unified_final.pt")
    if os.path.exists(unified_path):
        unified.load(unified_path)
        print("  Loaded unified")
    else:
        print("  WARNING: unified_final.pt not found")

    results = []
    for opp_name, opp_agent in sub_agents.items():
        print(f"\nRunning: Unified vs {opp_name} ({num_hands} hands)...",
              flush=True)
        rews = play_match_unified(env, unified, opp_agent, num_hands)
        bb   = bb_per_100(rews)
        wr   = winrate(rews)
        print(f"  BB/100={bb:+6.1f}  WinRate={wr:.1%}")
        results.append(bb)

    print("\n" + "-" * 60)
    avg = np.mean(results)
    print(f"  Unified overall avg BB/100: {avg:+.1f}")
    print("-" * 60)


def evaluate_meta(checkpoint_dir, num_hands=1000):
    if not _HAS_META:
        print("meta_agent not available, skipping.")
        return
    print("\n" + "=" * 60)
    print("META AGENT EVALUATION")
    print("=" * 60)
    env        = PokerEnv()
    sub_agents = load_sub_agents(
        checkpoint_dir, env.observation_size, env.action_size)
    meta = MetaAgent(
        obs_size    = env.observation_size,
        action_size = env.action_size,
        sub_agents  = sub_agents,
        mode        = "soft",
    )
    meta_path = os.path.join(checkpoint_dir, "meta_final.pt")
    if os.path.exists(meta_path):
        meta.load(meta_path)
        print("  Loaded meta")
    else:
        print("  WARNING: meta_final.pt not found")
    for opp_name, opp_agent in sub_agents.items():
        print(f"\nRunning: Meta vs {opp_name} ({num_hands} hands)...",
              flush=True)
        rews, weights = play_match_meta(env, meta, opp_agent, num_hands)
        bb  = bb_per_100(rews)
        wr  = winrate(rews)
        avg_w = np.mean(weights, axis=0) if weights else np.ones(4) / 4
        print(f"  BB/100={bb:+6.1f}  WinRate={wr:.1%}")
        print("  Avg Gating Weights:")
        for ename, w in zip(MetaAgent.EXPERT_NAMES, avg_w):
            bar = '#' * int(w * 30)
            print(f"    {ename:15s}: {w:.3f}  |{bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--num-hands",      type=int, default=1000)
    parser.add_argument(
        "--mode",
        choices=["sub", "meta", "unified", "all"],
        default="all",
    )
    args = parser.parse_args()
    if args.mode in ("sub", "all"):
        evaluate_subagents(args.checkpoint_dir, args.num_hands)
    if args.mode in ("meta", "all"):
        evaluate_meta(args.checkpoint_dir, args.num_hands)
    if args.mode in ("unified", "all"):
        evaluate_unified(args.checkpoint_dir, args.num_hands)


if __name__ == "__main__":
    main()

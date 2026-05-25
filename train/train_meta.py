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
    "aggressive": AggressiveAgent,
    "conservative": ConservativeAgent,
    "strategic": StrategicAgent,
    "deceptive": DeceptiveAgent,
}


def load_sub_agents(checkpoint_dir, obs_size, action_size):
    agents = {}
    for name, cls in SUB_AGENT_CLASSES.items():
        agent = cls(obs_size, action_size)
        path = os.path.join(checkpoint_dir, f"{name}_final.pt")
        if os.path.exists(path):
            agent.load(path)
            print(f"  Loaded {name}")
        else:
            print(f"  WARNING: {path} not found")
        agents[name] = agent
    return agents


def pretrain_gating(meta, battle_data_path, obs_size, epochs=10):
    """
    從 battle logs 預訓練 gating network。
    修正：
    - 每筆記錄重新建立對手風格 embedding（用 agent_a/b 的風格類型手工編碼）
    - 目標：當對手是 aggressive 時 label=conservative/strategic/deceptive
              當對手是 conservative 時 label=aggressive/deceptive
              用 winner 的對手型案來訓練
    """
    STYLE_TO_IDX = {
        "aggressive": 0, "conservative": 1,
        "strategic": 2, "deceptive": 3
    }
    # 對手風格的手工 embedding（訓練用）
    OPP_STYLE_EMBEDDINGS = {
        "aggressive":   np.array([0.8, 0.7, 0.9, 0.1], dtype=np.float32),
        "conservative": np.array([0.2, 0.1, 0.1, 0.8], dtype=np.float32),
        "strategic":    np.array([0.5, 0.4, 0.5, 0.4], dtype=np.float32),
        "deceptive":    np.array([0.5, 0.3, 0.6, 0.3], dtype=np.float32),
    }

    with open(battle_data_path, "rb") as f:
        records = pickle.load(f)

    print(f"Pretraining from {len(records)} battle records...")
    for epoch in range(epochs):
        random.shuffle(records)
        total_loss = 0
        count = 0
        for rec in records:
            if not rec.get("obs"):
                continue

            # 謎与方對手的風格 = 對手方的 agent_b 風格
            final_rew_a = rec["rewards_a"][-1] if rec["rewards_a"] else 0.0
            winner = rec["agent_a"] if final_rew_a > 0 else rec["agent_b"]
            opponent = rec["agent_b"] if final_rew_a > 0 else rec["agent_a"]

            # 目標：贏的那方的 expert idx
            best_expert = STYLE_TO_IDX.get(winner, 0)
            opp_emb = OPP_STYLE_EMBEDDINGS.get(opponent,
                np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32))

            obs_t = torch.FloatTensor(rec["obs"])
            opp_t = torch.FloatTensor(opp_emb).unsqueeze(0).expand(
                len(obs_t), -1)
            target = torch.zeros(len(obs_t), dtype=torch.long).fill_(best_expert)

            weights = meta.gating(obs_t, opp_t)
            loss = torch.nn.functional.cross_entropy(weights, target)
            entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()
            total = loss - 0.1 * entropy

            meta.optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(meta.gating.parameters(), 0.5)
            meta.optimizer.step()
            total_loss += loss.item()
            count += 1

        print(f"  Pretrain epoch {epoch+1}/{epochs}, loss: {total_loss/max(count,1):.4f}")


def ppo_meta_update(meta, sub_agents, expert_names, obs_list,
                    opp_styles, actions, old_logps, rewards, values,
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
    obs_t = torch.FloatTensor(np.array(obs_list, dtype=np.float32))
    opp_t = torch.FloatTensor(np.array(opp_styles, dtype=np.float32))
    act_t = torch.LongTensor(actions)
    old_logp_t = torch.FloatTensor(old_logps)
    adv_t = returns_t - torch.FloatTensor(values)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    weights = meta.gating(obs_t, opp_t)  # [N, 4]
    gating_entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()

    all_expert_probs = []
    for name in expert_names:
        agent = sub_agents[name]
        probs_list = [agent.get_action_probs(ob, list(range(meta.action_size)))
                      for ob in obs_list]
        all_expert_probs.append(torch.FloatTensor(np.array(probs_list)))
    expert_stack = torch.stack(all_expert_probs, dim=1)  # [N, 4, 8]
    mixed_probs = (weights.unsqueeze(2) * expert_stack).sum(dim=1)  # [N, 8]
    mixed_probs = mixed_probs.clamp(min=1e-8)
    mixed_probs = mixed_probs / mixed_probs.sum(dim=-1, keepdim=True)

    dist = torch.distributions.Categorical(probs=mixed_probs)
    new_logp = dist.log_prob(act_t)
    ratio = (new_logp - old_logp_t).exp()
    surr = torch.min(ratio * adv_t,
                     ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_t)
    loss = -surr.mean() - 0.05 * gating_entropy

    meta.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(meta.gating.parameters(), 0.5)
    meta.optimizer.step()


def train_meta(checkpoint_dir, battle_data_path, episodes, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    env = PokerEnv(num_players=2)
    sub_agents = load_sub_agents(
        checkpoint_dir, env.observation_size, env.action_size)
    meta = MetaAgent(
        obs_size=env.observation_size,
        action_size=env.action_size,
        sub_agents=sub_agents,
        mode="soft",
    )
    expert_names = MetaAgent.EXPERT_NAMES
    opp_list = list(sub_agents.values())
    opp_names = list(sub_agents.keys())

    if os.path.exists(battle_data_path):
        print("Pre-training gating network from battle logs...")
        pretrain_gating(meta, battle_data_path, env.observation_size, epochs=10)
    else:
        print(f"WARNING: {battle_data_path} not found, skipping pretrain")

    print(f"Fine-tuning Meta Agent for {episodes} episodes "
          f"(vs all sub-agents, round-robin)...")
    total_steps = 0
    pbar = tqdm(total=episodes)

    opp_idx = 0
    while total_steps < episodes:
        # 輪流對戰所有子模型，而不是僅對 random
        opp_agent = opp_list[opp_idx % len(opp_list)]
        opp_name  = opp_names[opp_idx % len(opp_names)]
        opp_idx  += 1

        meta.reset_opponent_tracker()
        obs = env.reset()
        ep_obs, ep_opp, ep_acts, ep_rews, ep_logps, ep_vals = \
            [], [], [], [], [], []
        done = False

        while not done:
            current = env.state.current_player
            legal   = env.get_legal_actions()

            if current == 0:
                action, logp, value, _ = meta.select_action(obs, legal)
                next_obs, _, done, _ = env.step(action)
                ep_obs.append(obs)
                ep_opp.append(meta.opp_tracker.get_embedding())
                ep_acts.append(action)
                ep_logps.append(logp)
                ep_vals.append(value)
                ep_rews.append(
                    float(env.state.get_reward(0)) if done else 0.0)
            else:
                # 對手用對應的子模型行動（而非 random）
                opp_action, _, _ = opp_agent.select_action(obs, legal)
                street = getattr(env.state, "street", 0)
                faced_raise = (opp_action >= 2)
                meta.observe_opponent(opp_action, street, faced_raise)
                next_obs, _, done, _ = env.step(opp_action)
                if done and ep_obs:
                    ep_rews[-1] = float(env.state.get_reward(0))

            obs = next_obs
            total_steps += 1

        if len(ep_obs) > 1:
            ppo_meta_update(meta, sub_agents, expert_names,
                            ep_obs, ep_opp, ep_acts, ep_logps,
                            ep_rews, ep_vals)

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

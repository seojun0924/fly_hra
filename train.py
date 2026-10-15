import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

import numpy as np
import gymnasium as gym
import torch

from agents import make_baseline_agent, make_fly_hra_agent
from reward_wrapper import ChannelDecomposedRewardWrapper


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def set_seed(seed, env):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)


def build_env(env_id, agent_type):
    env = gym.make(env_id)
    if agent_type == "fly_hra":
        if env_id not in ("LunarLander-v2", "LunarLander-v3"):
            raise ValueError("incorrect call")
        env = ChannelDecomposedRewardWrapper(env)
    return env


def run(env_id, agent_type, seed, episodes, target_sync_every, log_dir,
        max_steps_per_ep=1000, total_steps=None, overwrite=False):
    os.makedirs(log_dir, exist_ok=True)
    run_name = f"{env_id}_{agent_type}_seed{seed}"
    log_path = os.path.join(log_dir, f"{run_name}.csv")
    console_log_path = os.path.join(log_dir, f"{run_name}.console.log")
    meta_path = os.path.join(log_dir, f"{run_name}.meta.json")

    if os.path.exists(log_path) and not overwrite:
        raise SystemExit(
            f"attempted to overwrite while overwrite flag is not set"
        )

    console_file = open(console_log_path, "a", encoding="utf-8")
    real_stdout = sys.stdout
    sys.stdout = Tee(real_stdout, console_file)

    start_ts = time.time()
    start_iso = datetime.now(timezone.utc).astimezone().isoformat()
    ep = 0
    meta = None

    try:
        print(f"\n=== run start {start_iso} ({run_name}) ===")
        env = build_env(env_id, agent_type)
        set_seed(seed, env)

        state_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n

        if agent_type == "baseline":
            agent = make_baseline_agent(state_dim, n_actions)
        elif agent_type == "fly_hra":
            agent = make_fly_hra_agent(state_dim, n_actions, channel_weights=(1.0, 1.0, 1.0))
        else:
            raise ValueError(agent_type)

        n_heads = agent.n_heads

        header = ["episode", "return", "steps", "epsilon", "total_steps", "loss_mean"]
        if agent_type == "fly_hra":
            header += ["loss_ch_A", "loss_ch_B", "loss_ch_C", "return_ch_A", "return_ch_B", "return_ch_C"]

        stop_reason = "unknown"
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            while True:
                if episodes is not None and ep >= episodes:
                    stop_reason = "episodes_reached"
                    break
                if total_steps is not None and agent.step_count >= total_steps:
                    stop_reason = "total_steps_reached"
                    break

                obs, info = env.reset()
                done = False
                ep_return = 0.0
                ep_steps = 0
                loss_sum, loss_count = 0.0, 0
                per_channel_loss_sum = [0.0] * n_heads
                per_channel_return = [0.0] * n_heads

                while not done and ep_steps < max_steps_per_ep:
                    action = agent.act(obs)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    if agent_type == "fly_hra":
                        ch = info["reward_channels"]
                        reward_for_buffer = [ch["A"], ch["B"], ch["C"]]
                        assert abs(sum(reward_for_buffer) - info["reward_original"]) < 1e-3, (
                            f"channel sum ({sum(reward_for_buffer):.4f}) != "
                            f"original reward ({info['reward_original']:.4f})"
                        )
                        for i, v in enumerate(reward_for_buffer):
                            per_channel_return[i] += v
                    else:
                        reward_for_buffer = reward

                    agent.remember(obs, action, reward_for_buffer, next_obs, float(done))
                    result = agent.update()
                    if result is not None:
                        loss_value, per_channel_loss = result
                        loss_sum += loss_value
                        loss_count += 1
                        for i, v in enumerate(per_channel_loss):
                            per_channel_loss_sum[i] += v

                    obs = next_obs
                    ep_return += reward
                    ep_steps += 1

                if ep % target_sync_every == 0:
                    agent.sync_target()

                loss_mean = (loss_sum / loss_count) if loss_count else None
                row = [ep, ep_return, ep_steps, agent.epsilon(), agent.step_count, loss_mean]
                if agent_type == "fly_hra":
                    per_channel_loss_mean = [
                        (s / loss_count) if loss_count else None for s in per_channel_loss_sum
                    ]
                    row += per_channel_loss_mean + per_channel_return
                writer.writerow(row)

                if ep % 20 == 0:
                    print(f"[{agent_type} seed={seed}] ep {ep:4d} return={ep_return:8.2f} "
                          f"eps={agent.epsilon():.3f} total_steps={agent.step_count}")
                ep += 1

        env.close()
        elapsed_seconds = time.time() - start_ts
        print(f"complete: {log_path} (stopped by: {stop_reason}, total episode {ep}, total step {agent.step_count}, "
              f"elapsed {elapsed_seconds/60:.1f}분)")
        meta = {
            "run_name": run_name,
            "env": env_id,
            "agent": agent_type,
            "seed": seed,
            "episodes_requested": episodes,
            "total_steps_requested": total_steps,
            "stop_reason": stop_reason,
            "final_episode": ep,
            "final_total_steps": agent.step_count,
            "start_time": start_iso,
            "end_time": datetime.now(timezone.utc).astimezone().isoformat(),
            "elapsed_seconds": round(elapsed_seconds, 1),
            "log_csv": os.path.basename(log_path),
            "console_log": os.path.basename(console_log_path),
        }
        return meta
    except BaseException as exc:
        elapsed_seconds = time.time() - start_ts
        error_text = f"{type(exc).__name__}: {exc}"
        print(f"run fail: {run_name} (elapsed {elapsed_seconds/60:.1f}분) {error_text}")
        meta = {
            "run_name": run_name,
            "env": env_id,
            "agent": agent_type,
            "seed": seed,
            "episodes_requested": episodes,
            "total_steps_requested": total_steps,
            "stop_reason": "error",
            "error": error_text,
            "final_episode": ep,
            "start_time": start_iso,
            "end_time": datetime.now(timezone.utc).astimezone().isoformat(),
            "elapsed_seconds": round(elapsed_seconds, 1),
            "console_log": os.path.basename(console_log_path),
        }
        raise
    finally:
        if meta is not None:
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump(meta, mf, ensure_ascii=False, indent=2)
        sys.stdout = real_stdout
        console_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="LunarLander-v3")
    parser.add_argument("--agent", choices=["baseline", "fly_hra"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--target-sync-every", type=int, default=10)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.env, args.agent, args.seed, args.episodes, args.target_sync_every, args.log_dir,
        total_steps=args.total_steps, overwrite=args.overwrite)

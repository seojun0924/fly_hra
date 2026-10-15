import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for _font_name in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    if any(_font_name.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _font_name
        break
plt.rcParams["axes.unicode_minus"] = False

SOLVED_SCORE = 200.0
CONVERGENCE_THRESHOLD = 0.9 * SOLVED_SCORE
MOVING_AVG_WINDOW = 100
STABILITY_WINDOW = 100  # 마지막 N 에피소드


def load_runs(log_dir, env_id, agent_type):
    paths = sorted(glob.glob(os.path.join(log_dir, f"{env_id}_{agent_type}_seed*.csv")))
    return [pd.read_csv(p) for p in paths]


def convergence_episode(returns):
    moving_avg = returns.rolling(MOVING_AVG_WINDOW, min_periods=1).mean()
    reached = moving_avg[moving_avg >= CONVERGENCE_THRESHOLD]
    return int(reached.index[0]) if len(reached) else None


def sample_efficiency_auc(df):
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2.0 renamed trapz
    if "total_steps" in df.columns:
        return float(trapezoid(df["return"].values, x=df["total_steps"].values))
    print("W: old log. continuing")
    return float(trapezoid(df["return"].values))


def stability_std(returns):
    tail = returns.iloc[-STABILITY_WINDOW:]
    return float(tail.std())


def summarize(log_dir, env_id, agent_type):
    runs = load_runs(log_dir, env_id, agent_type)
    if not runs:
        print(f"W: {agent_type} has no log ({log_dir}/{env_id}_{agent_type}_seed*.csv)")
        return None

    conv, auc, std = [], [], []
    for df in runs:
        c = convergence_episode(df["return"])
        conv.append(c if c is not None else len(df))
        auc.append(sample_efficiency_auc(df))
        std.append(stability_std(df["return"]))

    print(f"\n== {agent_type} (n={len(runs)} seeds) ==")
    print(f"  수렴 속도(에피소드, 낮을수록 좋음): 평균 {np.mean(conv):.1f} ± {np.std(conv):.1f}")
    print(f"  샘플 효율성(AUC, 높을수록 좋음): 평균 {np.mean(auc):.1f} ± {np.std(auc):.1f}")
    print(f"  안정성(마지막 {STABILITY_WINDOW}ep 표준편차, 낮을수록 좋음): 평균 {np.mean(std):.1f} ± {np.std(std):.1f}")
    return {"convergence": conv, "auc": auc, "stability": std}


def plot_learning_curves_by_steps(log_dir, env_id, out_path):
    plt.figure(figsize=(8, 5))
    plotted_any = False
    for agent_type, color in [("baseline", "tab:gray"), ("fly_hra", "tab:orange")]:
        runs = load_runs(log_dir, env_id, agent_type)
        runs = [df for df in runs if "total_steps" in df.columns]
        if not runs:
            continue
        common_max_steps = min(df["total_steps"].iloc[-1] for df in runs)
        grid = np.linspace(0, common_max_steps, 500)
        curves = []
        for df in runs:
            smoothed = df["return"].rolling(MOVING_AVG_WINDOW, min_periods=1).mean()
            curves.append(np.interp(grid, df["total_steps"].values, smoothed.values))
        curves = np.stack(curves)
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        plt.plot(grid, mean, label=agent_type, color=color)
        plt.fill_between(grid, mean - std, mean + std, alpha=0.15, color=color)
        plotted_any = True

    if not plotted_any:
        plt.close()
        print(f"skipped per-step info; perhaps old log?")
        return

    plt.axhline(SOLVED_SCORE, linestyle="--", color="black", linewidth=0.8, label="solved (200)")
    plt.xlabel("total steps (cumulative)")
    plt.ylabel("return")
    plt.legend()
    plt.title(f"{env_id}: baseline vs fly_hra")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"png, per-step saved at: {out_path}")


def plot_learning_curves(log_dir, env_id, out_path):
    plt.figure(figsize=(8, 5))
    for agent_type, color in [("baseline", "tab:gray"), ("fly_hra", "tab:orange")]:
        runs = load_runs(log_dir, env_id, agent_type)
        if not runs:
            continue
        min_len = min(len(df) for df in runs)
        curves = np.stack([df["return"].values[:min_len] for df in runs])
        mean = curves.mean(axis=0)
        mean_smooth = pd.Series(mean).rolling(MOVING_AVG_WINDOW, min_periods=1).mean()
        std = curves.std(axis=0)
        x = np.arange(min_len)
        plt.plot(x, mean_smooth, label=agent_type, color=color)
        plt.fill_between(x, mean_smooth - std, mean_smooth + std, alpha=0.15, color=color)

    plt.axhline(SOLVED_SCORE, linestyle="--", color="black", linewidth=0.8, label="solved (200)")
    plt.xlabel("episode")
    plt.ylabel("return")
    plt.legend()
    plt.title(f"{env_id}: baseline vs fly_hra")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"png, per-episode saved at: {out_path}")


def plot_channel_diagnostics(log_dir, env_id, out_path):
    runs = load_runs(log_dir, env_id, "fly_hra")
    if not runs or "loss_ch_A" not in runs[0].columns:
        print("skip channel, since no fit was found (perhaps the log is legacy?)")
        return
    min_len = min(len(df) for df in runs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ch, color in [("A", "tab:blue"), ("B", "tab:red"), ("C", "tab:green")]:
        loss_curves = np.stack([df[f"loss_ch_{ch}"].values[:min_len] for df in runs])
        ret_curves = np.stack([df[f"return_ch_{ch}"].values[:min_len] for df in runs])
        loss_mean = pd.Series(loss_curves.mean(axis=0)).rolling(MOVING_AVG_WINDOW, min_periods=1).mean()
        ret_mean = pd.Series(ret_curves.mean(axis=0)).rolling(MOVING_AVG_WINDOW, min_periods=1).mean()
        axes[0].plot(loss_mean, label=f"channel {ch}", color=color)
        axes[1].plot(ret_mean, label=f"channel {ch}", color=color)
    axes[0].set_title("TD-error loss, per-channel (100ep avg)")
    axes[0].set_xlabel("episode")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[1].set_title("accumulated reward, per-channel (100ep avg)")
    axes[1].set_xlabel("episode")
    axes[1].set_ylabel("return (channel)")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"plot saved at {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--env", default="LunarLander-v3")
    args = parser.parse_args()

    summarize(args.log_dir, args.env, "baseline")
    summarize(args.log_dir, args.env, "fly_hra")
    plot_learning_curves_by_steps(
        args.log_dir, args.env, os.path.join(args.log_dir, f"{args.env}_learning_curves_by_steps.png")
    )
    plot_learning_curves(
        args.log_dir, args.env, os.path.join(args.log_dir, f"{args.env}_learning_curves_by_episode.png")
    )
    plot_channel_diagnostics(args.log_dir, args.env, os.path.join(args.log_dir, f"{args.env}_channel_diagnostics.png"))

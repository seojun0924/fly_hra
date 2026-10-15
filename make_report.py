import argparse
import glob
import json
import os
from datetime import datetime

import numpy as np

import evaluate_metrics as em


def load_meta(log_dir, env_id):
    paths = sorted(glob.glob(os.path.join(log_dir, f"{env_id}_*_seed*.meta.json")))
    metas = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            metas.append(json.load(f))
    return metas


def format_elapsed(seconds):
    if seconds is None:
        return "?"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def build_run_table(metas):
    if not metas:
        return "  (no records yet)\n"
    lines = []
    header = f"  {'run_name':45s} {'stop_reason':20s} {'ep':>6s} {'steps':>9s} {'소요시간':>14s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for m in metas:
        name = m.get("run_name", "?")
        reason = m.get("stop_reason", "?")
        ep = m.get("final_episode", "?")
        steps = m.get("final_total_steps", "?")
        elapsed = format_elapsed(m.get("elapsed_seconds"))
        lines.append(f"  {name:45s} {reason:20s} {str(ep):>6s} {str(steps):>9s} {elapsed:>14s}")
    return "\n".join(lines) + "\n"


def build_summary_section(agent_type, log_dir, env_id):
    result = em.summarize(log_dir, env_id, agent_type)
    if result is None:
        return f"  {agent_type}: no log found)\n"
    conv, auc, std = result["convergence"], result["auc"], result["stability"]
    lines = [
        f"  == {agent_type} (n={len(conv)} seeds) ==",
        f"    수렴 속도(에피소드, 낮을수록 좋음): 평균 {np.mean(conv):.1f} ± {np.std(conv):.1f}",
        f"    샘플 효율성(AUC, 높을수록 좋음): 평균 {np.mean(auc):.1f} ± {np.std(auc):.1f}",
        f"    안정성(마지막 {em.STABILITY_WINDOW}ep 표준편차, 낮을수록 좋음): 평균 {np.mean(std):.1f} ± {np.std(std):.1f}",
    ]
    return "\n".join(lines) + "\n"


def build_report(log_dir, env_id):
    metas = load_meta(log_dir, env_id)
    now = datetime.now().astimezone().isoformat(timespec="seconds")

    parts = []
    parts.append("=" * 70)
    parts.append(f"  report for {env_id}")
    parts.append(f"  timestamp: {now}")
    parts.append(f"  log-dir: {os.path.abspath(log_dir)}")
    parts.append("=" * 70)
    parts.append("")

    parts.append("[1] run record, per-seed")
    parts.append(build_run_table(metas))
    n_error = sum(1 for m in metas if m.get("stop_reason") == "error")
    if n_error:
        parts.append(f"  failed {n_error} seeds. check column stop_reason=error: "
                     f"check field of error at meta.json.")
        parts.append("")
    total_elapsed = sum(m.get("elapsed_seconds", 0) or 0 for m in metas)
    parts.append(f"  total run {len(metas)}, elapsed time {format_elapsed(total_elapsed)}")
    parts.append("")

    parts.append("[2] metrics")
    parts.append(build_summary_section("baseline", log_dir, env_id))
    parts.append(build_summary_section("fly_hra", log_dir, env_id))
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--env", default="LunarLander-v3")
    parser.add_argument("--out", default=None,
                        help="save dir (defaults to: <log-dir>/REPORT_<env>.txt)")
    args = parser.parse_args()

    report_text = build_report(args.log_dir, args.env)
    print(report_text)

    out_path = args.out or os.path.join(args.log_dir, f"REPORT_{args.env}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nsaved report at: {out_path}")

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def config_name(cfg: dict) -> str:
    parts = [cfg.get("scope", "?"), cfg.get("pretrain", "?")]
    if cfg.get("norm") != "instance":
        parts.append(f"norm-{cfg.get('norm')}")
    if not cfg.get("adabn"):
        parts.append("noadabn")
    if cfg.get("channels", "both") != "both":
        parts.append(cfg["channels"])
    if cfg.get("ms_label_budget") == "total":
        parts.append("total-budget")
    return "|".join(parts)


def load_runs(results_dir: Path) -> list[dict]:
    runs = []
    for path in sorted(results_dir.glob("results_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        cfg = data.get("config", {})
        data["_name"] = config_name(cfg)
        data["_seed"] = cfg.get("seed")
        data["_file"] = path.name
        runs.append(data)
    return runs


def fmt(mean, std, n):
    if mean is None:
        return "   -            "
    if n > 1 and std is not None:
        return f"{mean:.4f}+-{std:.4f}"
    return f"{mean:.4f}        "


def _stats(entries, key):
    vals = [e[key] for e in entries if e.get(key) is not None]
    if not vals:
        return None, None
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
    return float(np.mean(vals)), std


def aggregate(runs: list[dict], summary_key: str, columns, title: str) -> None:
    by_key = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for spc, s in run.get(summary_key, {}).items():
            by_key[run["_name"]][int(spc)].append(s)
    if not by_key:
        return

    for spc in sorted({spc for d in by_key.values() for spc in d}):
        print(f"\n=== {title}  spc={spc} ===")
        header = f"{'config':40s}  n  " + " ".join(f"{c[1]:16s}" for c in columns)
        print(header)
        print("-" * len(header))

        rows = []
        for name, per_spc in by_key.items():
            entries = per_spc.get(spc, [])
            if not entries:
                continue
            cells = [_stats(entries, key) for key, _ in columns]
            sort_by = cells[0][0] if cells[0][0] is not None else -1
            rows.append((sort_by, name, len(entries),
                         [fmt(m, s, len(entries)) for m, s in cells]))

        for _, name, n, cells in sorted(rows, reverse=True):
            print(f"{name:40s}  {n}  " + " ".join(cells))


def paired_cross_values(runs: list[dict], name: str, spc: int) -> dict:
    out = {}
    for run in runs:
        if run["_name"] != name:
            continue
        for train_load, d in run.get("single_source", {}).get(str(spc), {}).items():
            for test_load, acc in d.get("cross", {}).items():
                out[(run["_seed"], train_load, test_load)] = acc
    return out


def compare(runs: list[dict], sub_a: str, sub_b: str, spc: int) -> None:
    from scipy.stats import wilcoxon

    names = sorted({r["_name"] for r in runs})
    pick = lambda sub: [n for n in names if sub in n]
    for sub in (sub_a, sub_b):
        matches = pick(sub)
        if len(matches) != 1:
            print(f"\n[compare] '{sub}' matches {len(matches)} configs: {matches}")
            print("          Use a more specific substring.")
            return
    name_a, name_b = pick(sub_a)[0], pick(sub_b)[0]

    va = paired_cross_values(runs, name_a, spc)
    vb = paired_cross_values(runs, name_b, spc)
    keys = sorted(set(va) & set(vb))
    if len(keys) < 6:
        print(f"\n[compare] only {len(keys)} paired (seed, train, test) points; "
              "more seeds or load pairs are needed for a meaningful test.")
        return

    a = np.array([va[k] for k in keys])
    b = np.array([vb[k] for k in keys])
    stat, p = wilcoxon(a, b)
    print(f"\n[Wilcoxon signed-rank]  spc={spc}  n_pairs={len(keys)}")
    print(f"  A = {name_a}: cross mean {a.mean():.4f}")
    print(f"  B = {name_b}: cross mean {b.mean():.4f}")
    print(f"  delta = {a.mean() - b.mean():+.4f}   W={stat:.1f}   p={p:.2e}"
          f"   {'(significant at 0.05)' if p < 0.05 else '(not significant)'}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate result JSONs into mean+-std tables")
    ap.add_argument("--dir", type=str, default="results")
    ap.add_argument("--compare", type=str, nargs=2, metavar=("A", "B"), default=None,
                    help="Two config-name substrings to test against each other.")
    ap.add_argument("--spc", type=int, default=20,
                    help="Label budget used for --compare.")
    args = ap.parse_args()

    runs = load_runs(Path(args.dir))
    if not runs:
        raise SystemExit(f"No results_*.json under {args.dir}")
    print(f"Loaded {len(runs)} runs from {args.dir}")

    aggregate(runs, "single_source_summary",
              (("cross_load_avg", "cross_acc"),
               ("cross_load_f1_avg", "cross_f1"),
               ("same_load_avg", "same_acc")),
              "single-source")
    aggregate(runs, "leave_one_load_out_summary",
              (("target_acc_avg", "target_acc"),
               ("target_f1_avg", "target_f1"),
               ("source_acc_avg", "source_acc")),
              "leave-one-load-out")

    if args.compare:
        compare(runs, args.compare[0], args.compare[1], args.spc)


if __name__ == "__main__":
    main()

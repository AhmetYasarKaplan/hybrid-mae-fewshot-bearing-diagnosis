from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import train_pipeline as tp
from models import build_embedder


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_loads(args, label_to_int):
    """Load, split and normalise every operating condition once."""
    prepared = {}
    for load_id in args.loads:
        ld = tp.load_windows(
            args.data_dir, load_id,
            window_size=args.window_size, stride=args.stride,
            channels=args.channels,
            window_cache_dir=Path(args.window_cache_dir) if args.window_cache_dir else None,
            use_cache=not args.no_window_cache,
        )
        tr_idx, te_idx = tp.make_split(ld.y, args.test_size, args.seed)

        X_tr, X_te = ld.X[tr_idx], ld.X[te_idx]
        mean, std = tp.norm_stats(X_tr)

        y_tr = np.asarray([label_to_int[v] for v in ld.y[tr_idx]])
        y_te = np.asarray([label_to_int[v] for v in ld.y[te_idx]])
        m_idx, v_idx = tp.stratified_val_split(y_tr, args.val_split, args.seed)

        prepared[load_id] = {
            "X_tr": X_tr, "y_tr": y_tr,
            "X_te": X_te, "y_te": y_te,
            "mean": mean, "std": std,
            "m_idx": m_idx, "v_idx": v_idx,
            "in_channels": X_tr.shape[1],
            "classes": ld.classes,
            "fs": ld.fs,
        }
        print(f"  Load {load_id}: train {X_tr.shape} test {X_te.shape} "
              f"classes={ld.classes}")
    return prepared


def build_embedders(args, prepared, device):
    """Pretrain the autoencoders and return {load: frozen embedder}."""
    in_channels = next(iter(prepared.values()))["in_channels"]

    common = dict(pretrain=args.pretrain, con_weight=args.con_weight,
                  temperature=args.temperature, mask_ratio=args.mask_ratio,
                  epochs=args.ae_epochs, batch_size=args.batch_size,
                  lr=args.ae_lr, patience=args.ae_patience, device=device)

    if args.scope == "all_loads":
        X_domains = [tp.normalize(p["X_tr"][p["m_idx"]], p["mean"], p["std"], args.norm)
                     for p in prepared.values()]
        Xv = torch.cat([tp.normalize(p["X_tr"][p["v_idx"]], p["mean"], p["std"], args.norm)
                        for p in prepared.values()], dim=0)
        n_total = sum(len(X) for X in X_domains)
        print(f"\n[all_loads] pretraining one shared autoencoder on "
              f"{len(X_domains)} loads ({n_total} windows) ...")
        ae = tp.pretrain_ae(in_channels, args.latent, X_domains, Xv,
                            log_prefix="AE-ALL", **common)
        shared = build_embedder([ae])
        return {lid: shared for lid in prepared}

    if args.scope == "sensor_experts":
        if in_channels < 2:
            raise SystemExit("--scope sensor_experts needs --channels both; "
                             "with a single channel use --scope all_loads.")
        experts, slices = [], []
        for ch in range(in_channels):
            name = {0: "DE", 1: "FE"}.get(ch, f"ch{ch}")
            X_domains = [tp.normalize(p["X_tr"][p["m_idx"]], p["mean"], p["std"],
                                      args.norm)[:, ch:ch + 1]
                         for p in prepared.values()]
            Xv = torch.cat([tp.normalize(p["X_tr"][p["v_idx"]], p["mean"], p["std"],
                                         args.norm)[:, ch:ch + 1]
                            for p in prepared.values()], dim=0)
            n_total = sum(len(X) for X in X_domains)
            print(f"\n[sensor_experts] pretraining the {name} expert "
                  f"({n_total} windows, all loads pooled) ...")
            ae = tp.pretrain_ae(1, args.latent, X_domains, Xv,
                                log_prefix=f"AE-{name}", **common)
            experts.append(ae)
            slices.append([ch])
        fused = build_embedder(experts, channel_slices=slices)
        print(f"  fused embedding dim = {fused.embedding_dim} "
              f"({fused.num_experts} sensors x {fused.latent_channels})")
        return {lid: fused for lid in prepared}

    raise ValueError(f"Unknown scope {args.scope!r}")


def run_single_source(args, prepared, embedders, num_classes, device):
    """Train the head on one load, evaluate on that load and on every other one."""
    results = {}
    adapt_cache = {}

    def adapted_embedder(train_load, test_load):
        # adapted on the target's unlabelled train pool, never on its test windows
        if not args.adabn:
            return embedders[train_load]
        key = (id(embedders[train_load]), test_load)
        if key not in adapt_cache:
            q = prepared[test_load]
            X_pool = tp.normalize(q["X_tr"], q["mean"], q["std"], args.norm)
            adapt_cache[key] = tp.adapt_batchnorm(embedders[train_load], X_pool, device)
        return adapt_cache[key]

    for spc in args.spc:
        results[spc] = {}
        for train_load, p in prepared.items():
            ym = p["y_tr"][p["m_idx"]]
            lab_idx = tp.labeled_subset(ym, spc, args.seed)
            X_lab = tp.normalize(p["X_tr"][p["m_idx"]][lab_idx],
                                 p["mean"], p["std"], args.norm)
            y_lab = torch.from_numpy(ym[lab_idx]).long()
            X_val = tp.normalize(p["X_tr"][p["v_idx"]], p["mean"], p["std"], args.norm)
            y_val = torch.from_numpy(p["y_tr"][p["v_idx"]]).long()

            head = tp.train_fusion_head(
                embedders[train_load], X_lab, y_lab, X_val, y_val, num_classes,
                epochs=args.clf_epochs, lr=args.clf_lr,
                patience=args.clf_patience, device=device,
            )

            X_te = tp.normalize(p["X_te"], p["mean"], p["std"], args.norm)
            same = tp.evaluate(p["y_te"],
                               tp.predict(embedders[train_load], head, X_te, device))
            print(f"\n  spc={spc}  train=L{train_load}")
            print(f"    L{train_load}->L{train_load} (same)   "
                  f"acc={same['accuracy']:.4f}  macro_f1={same['macro_f1']:.4f}")

            cross, cross_f1 = {}, {}
            for test_load, q in prepared.items():
                if test_load == train_load:
                    continue
                Xq = tp.normalize(q["X_te"], q["mean"], q["std"], args.norm)
                emb_t = adapted_embedder(train_load, test_load)
                m = tp.evaluate(q["y_te"], tp.predict(emb_t, head, Xq, device))
                cross[test_load] = m["accuracy"]
                cross_f1[test_load] = m["macro_f1"]
                print(f"    L{train_load}->L{test_load} (cross)  "
                      f"acc={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}")

            results[spc][train_load] = {"same": same, "cross": cross,
                                        "cross_f1": cross_f1}
            cross_avg = float(np.mean(list(cross.values()))) if cross else float("nan")
            print(f"    cross_avg={cross_avg:.4f}")
    return results


def run_leave_one_load_out(args, prepared, embedders, num_classes, device):
    """Train the head on K-1 loads, evaluate on the held-out one."""
    results = {}
    adapt_cache = {}
    loads = sorted(prepared.keys())

    for spc in args.spc:
        results[spc] = {}
        for target in loads:
            sources = [l for l in loads if l != target]
            embedder = embedders[target]

            X_parts, y_parts, Xv_parts, yv_parts = [], [], [], []
            for s in sources:
                p = prepared[s]
                ym = p["y_tr"][p["m_idx"]]
                idx = (tp.labeled_subset(ym, spc, args.seed)
                       if args.ms_label_budget == "per_source" else np.arange(len(ym)))
                X_parts.append(tp.normalize(p["X_tr"][p["m_idx"]][idx],
                                            p["mean"], p["std"], args.norm))
                y_parts.append(ym[idx])
                Xv_parts.append(tp.normalize(p["X_tr"][p["v_idx"]],
                                             p["mean"], p["std"], args.norm))
                yv_parts.append(p["y_tr"][p["v_idx"]])

            X_lab = torch.cat(X_parts, dim=0)
            y_all = np.concatenate(y_parts)
            if args.ms_label_budget == "total":
                keep = tp.labeled_subset(y_all, spc, args.seed)
                X_lab, y_all = X_lab[keep], y_all[keep]
            y_lab = torch.from_numpy(y_all).long()
            X_val = torch.cat(Xv_parts, dim=0)
            y_val = torch.from_numpy(np.concatenate(yv_parts)).long()

            head = tp.train_fusion_head(
                embedder, X_lab, y_lab, X_val, y_val, num_classes,
                epochs=args.clf_epochs, lr=args.clf_lr,
                patience=args.clf_patience, device=device,
            )

            print(f"\n  spc={spc}  sources={sources} -> target=L{target}"
                  f"  ({len(y_lab)} labelled windows)")

            src_accs = {}
            for s in sources:
                q = prepared[s]
                Xq = tp.normalize(q["X_te"], q["mean"], q["std"], args.norm)
                src_accs[s] = tp.evaluate(
                    q["y_te"], tp.predict(embedder, head, Xq, device))["accuracy"]
            src_avg = float(np.mean(list(src_accs.values()))) if src_accs else float("nan")
            print(f"    source-domain avg acc={src_avg:.4f}  {src_accs}")

            emb_t = embedder
            if args.adabn:
                key = (id(embedder), target)
                if key not in adapt_cache:
                    q = prepared[target]
                    X_pool = tp.normalize(q["X_tr"], q["mean"], q["std"], args.norm)
                    adapt_cache[key] = tp.adapt_batchnorm(embedder, X_pool, device)
                emb_t = adapt_cache[key]

            q = prepared[target]
            Xq = tp.normalize(q["X_te"], q["mean"], q["std"], args.norm)
            m = tp.evaluate(q["y_te"], tp.predict(emb_t, head, Xq, device))
            print(f"    -> L{target} (unseen)  acc={m['accuracy']:.4f}  "
                  f"macro_f1={m['macro_f1']:.4f}")

            results[spc][target] = {
                "target": m, "sources": sources,
                "source_acc": src_accs, "source_avg": src_avg,
                "n_labelled": int(len(y_lab)),
            }

        tgt = [d["target"]["accuracy"] for d in results[spc].values()]
        print(f"\n  spc={spc}  leave-one-load-out avg = {float(np.mean(tgt)):.4f}")
    return results


def run_baselines(args, prepared, num_classes, device):
    from torch.utils.data import DataLoader, TensorDataset

    from feature_extraction import extract_features_batch
    from models import CNN1D, create_svm_model, create_xgboost_model
    from training import train_cnn, get_cnn_predictions

    out = {"cnn1d": {}, "svm": {}, "xgboost": {}}
    avg = lambda d: float(np.mean(list(d.values()))) if d else float("nan")

    test_feats = {tl: extract_features_batch(q["X_te"], fs=q["fs"],
                                             feature_set=args.feature_set)
                  for tl, q in prepared.items()}

    for spc in args.spc:
        for name in out:
            out[name][spc] = {}
        for train_load, p in prepared.items():
            print(f"\n  [baselines] spc={spc}  train=L{train_load}")
            ym = p["y_tr"][p["m_idx"]]
            lab_idx = tp.labeled_subset(ym, spc, args.seed)
            y_lab = ym[lab_idx]

            X_raw_lab = p["X_tr"][p["m_idx"]][lab_idx]
            X_lab = tp.normalize(X_raw_lab, p["mean"], p["std"], args.norm)
            X_val = tp.normalize(p["X_tr"][p["v_idx"]], p["mean"], p["std"], args.norm)
            y_val = torch.from_numpy(p["y_tr"][p["v_idx"]]).long()
            yl = torch.from_numpy(y_lab).long()

            cnn = CNN1D(num_classes=num_classes, in_channels=p["in_channels"]).to(device)
            loader = DataLoader(TensorDataset(X_lab, yl),
                                batch_size=min(args.batch_size, len(yl)), shuffle=True)
            train_cnn(cnn, loader, X_lab, yl, X_val=X_val, y_val=y_val,
                      epochs=args.clf_epochs, device=device,
                      print_every=args.clf_epochs + 1)

            X_te = tp.normalize(p["X_te"], p["mean"], p["std"], args.norm)
            same_cnn = tp.evaluate(p["y_te"], get_cnn_predictions(cnn, X_te, device))
            cross_cnn = {}
            for tl, q in prepared.items():
                if tl == train_load:
                    continue
                Xq = tp.normalize(q["X_te"], q["mean"], q["std"], args.norm)
                cnn_t = cnn
                if args.adabn:
                    X_pool = tp.normalize(q["X_tr"], q["mean"], q["std"], args.norm)
                    cnn_t = tp.adapt_batchnorm(cnn, X_pool, device)
                cross_cnn[tl] = tp.evaluate(
                    q["y_te"], get_cnn_predictions(cnn_t, Xq, device))["accuracy"]
            out["cnn1d"][spc][train_load] = {"same": same_cnn, "cross": cross_cnn}
            print(f"    CNN1D    same_acc={same_cnn['accuracy']:.4f}  "
                  f"f1={same_cnn['macro_f1']:.4f}  cross_avg={avg(cross_cnn):.4f}")

            feat_tr = extract_features_batch(X_raw_lab, fs=p["fs"],
                                             feature_set=args.feature_set)
            for name, ctor in (("svm", create_svm_model),
                               ("xgboost", lambda: create_xgboost_model(num_classes))):
                try:
                    clf = ctor()
                    clf.fit(feat_tr, y_lab)
                    same = tp.evaluate(p["y_te"], clf.predict(test_feats[train_load]))
                    cross = {tl: tp.evaluate(q["y_te"],
                                             clf.predict(test_feats[tl]))["accuracy"]
                             for tl, q in prepared.items() if tl != train_load}
                    out[name][spc][train_load] = {"same": same, "cross": cross}
                    print(f"    {name.upper():8s} same_acc={same['accuracy']:.4f}  "
                          f"f1={same['macro_f1']:.4f}  cross_avg={avg(cross):.4f}")
                except Exception as exc:
                    out[name][spc][train_load] = {"error": str(exc)}
                    print(f"    {name.upper():8s} ERROR: {exc}")
    return out


def print_cross_matrices(results, loads):
    for spc, per_load in results.items():
        print(f"\n[accuracy matrix]  spc={spc}  "
              f"(rows=train, cols=test, diagonal=same-load)")
        print("        " + "".join(f"    L{t}   " for t in loads))
        for tr in loads:
            row = f"  L{tr}  "
            d = per_load.get(tr, {})
            for te in loads:
                v = (d.get("same", {}).get("accuracy") if te == tr
                     else d.get("cross", {}).get(te))
                row += f"  {v:.4f} " if v is not None else "     -   "
            print(row)


def summarise_single_source(results: dict) -> dict:
    summary = {}
    for spc, per_load in results.items():
        sames, crosses, crosses_f1 = [], [], []
        for d in per_load.values():
            sames.append(d["same"]["accuracy"])
            crosses.extend(d["cross"].values())
            crosses_f1.extend(d.get("cross_f1", {}).values())
        summary[str(spc)] = {
            "same_load_avg": float(np.mean(sames)) if sames else None,
            "cross_load_avg": float(np.mean(crosses)) if crosses else None,
            "cross_load_f1_avg": float(np.mean(crosses_f1)) if crosses_f1 else None,
        }
    return summary


def summarise_lolo(results: dict) -> dict:
    summary = {}
    for spc, per_target in results.items():
        accs = [d["target"]["accuracy"] for d in per_target.values()]
        f1s = [d["target"]["macro_f1"] for d in per_target.values()]
        srcs = [d["source_avg"] for d in per_target.values()]
        summary[str(spc)] = {
            "target_acc_avg": float(np.mean(accs)) if accs else None,
            "target_f1_avg": float(np.mean(f1s)) if f1s else None,
            "source_acc_avg": float(np.mean(srcs)) if srcs else None,
            "per_target": {str(t): d["target"]["accuracy"]
                           for t, d in per_target.items()},
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Few-shot cross-load bearing fault diagnosis on CWRU")
    ap.add_argument("--data-dir", type=str, default="data")
    ap.add_argument("--output-dir", type=str, default="results")
    ap.add_argument("--loads", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--channels", choices=["de", "fe", "both"], default="both",
                    help="drive-end, fan-end, or both accelerometers.")

    ap.add_argument("--scope", choices=["all_loads", "sensor_experts"],
                    default="sensor_experts",
                    help="one shared autoencoder, or one expert per input channel.")
    ap.add_argument("--pretrain", choices=["mae", "contrastive", "hybrid"],
                    default="hybrid",
                    help="masked reconstruction, NT-Xent, or both.")
    ap.add_argument("--con-weight", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.2)

    ap.add_argument("--protocol", choices=["single_source", "multi_source", "both"],
                    default="single_source",
                    help="train the head on one load, on K-1 loads, or report both.")
    ap.add_argument("--ms-label-budget", choices=["per_source", "total"],
                    default="per_source",
                    help="multi_source only: spc labels per source load, or spc in "
                         "total across the merged source pool.")

    ap.add_argument("--norm", choices=["dataset", "instance"], default="instance",
                    help="per-load statistics, or per-window standardisation.")
    ap.add_argument("--adabn", action="store_true",
                    help="re-estimate BatchNorm statistics on the target load's "
                         "unlabelled train pool before cross-load prediction.")

    ap.add_argument("--spc", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--window-size", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=1024)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--mask-ratio", type=float, default=0.35)
    ap.add_argument("--ae-epochs", type=int, default=120)
    ap.add_argument("--ae-lr", type=float, default=1e-3)
    ap.add_argument("--ae-patience", type=int, default=20)
    ap.add_argument("--clf-epochs", type=int, default=150)
    ap.add_argument("--clf-lr", type=float, default=1e-3)
    ap.add_argument("--clf-patience", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)

    ap.add_argument("--feature-set", choices=["standard", "envelope", "all"],
                    default="all", help="feature set for the SVM / XGBoost baselines.")
    ap.add_argument("--window-cache-dir", type=str, default=None)
    ap.add_argument("--no-window-cache", action="store_true")
    ap.add_argument("--baselines", action="store_true",
                    help="also train the CNN1D / SVM / XGBoost baselines.")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = (("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else args.device)

    banner = [
        f"loads={args.loads}  channels={args.channels}  "
        f"window={args.window_size}/{args.stride}",
        f"scope={args.scope}  pretrain={args.pretrain}  latent={args.latent}"
        + (f"  (con_weight={args.con_weight}, tau={args.temperature})"
           if args.pretrain != "mae" else ""),
        f"protocol={args.protocol}"
        + (f"  (ms_label_budget={args.ms_label_budget})"
           if args.protocol != "single_source" else ""),
        f"norm={args.norm}  adabn={args.adabn}  spc={args.spc}  "
        f"seed={args.seed}  device={device}",
    ]
    width = max(len(line) for line in banner) + 4
    print("=" * width)
    print("RUN CONFIG")
    for line in banner:
        print(f"  {line}")
    print("=" * width)

    classes = set()
    for lid in args.loads:
        ld = tp.load_windows(
            args.data_dir, lid,
            window_size=args.window_size, stride=args.stride,
            channels=args.channels,
            window_cache_dir=Path(args.window_cache_dir) if args.window_cache_dir else None,
            use_cache=not args.no_window_cache)
        classes.update(ld.classes)
    classes = sorted(classes)
    label_to_int = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")

    print("\nPreparing loads ...")
    prepared = prepare_loads(args, label_to_int)

    print("\nPretraining ...")
    embedders = build_embedders(args, prepared, device)

    payload = {"config": vars(args), "classes": classes}

    if args.protocol in {"single_source", "both"}:
        print("\nEvaluating single-source (train on one load, test on all) ...")
        res = run_single_source(args, prepared, embedders, num_classes, device)
        print_cross_matrices(res, sorted(prepared.keys()))
        payload["single_source"] = {str(spc): {str(tl): v for tl, v in per_load.items()}
                                    for spc, per_load in res.items()}
        payload["single_source_summary"] = summarise_single_source(res)

    if args.protocol in {"multi_source", "both"}:
        print("\nEvaluating leave-one-load-out ...")
        res = run_leave_one_load_out(args, prepared, embedders, num_classes, device)
        payload["leave_one_load_out"] = {str(spc): {str(t): v for t, v in per_t.items()}
                                         for spc, per_t in res.items()}
        payload["leave_one_load_out_summary"] = summarise_lolo(res)

    if args.baselines:
        print("\nRunning baselines (CNN1D / SVM / XGBoost) ...")
        base = run_baselines(args, prepared, num_classes, device)
        payload["baselines"] = {
            name: {str(spc): {str(tl): v for tl, v in per_load.items()}
                   for spc, per_load in by_spc.items()}
            for name, by_spc in base.items()
        }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extras = f"_{args.pretrain}"
    if args.adabn:
        extras += "_adabn"
    if args.protocol != "single_source":
        extras += f"_{args.protocol}_{args.ms_label_budget}"
    tag = f"{args.scope}_{args.channels}_{args.norm}{extras}_seed{args.seed}"
    out_path = out_dir / f"results_{tag}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for key, title in (("single_source_summary", "single-source"),
                       ("leave_one_load_out_summary", "leave-one-load-out")):
        if key in payload:
            print(f"\nSummary [{title}]: {json.dumps(payload[key], indent=2)}")
    print(f"Results -> {out_path}")


if __name__ == "__main__":
    main()

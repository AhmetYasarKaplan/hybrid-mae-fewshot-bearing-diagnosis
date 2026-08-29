from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from data_pipeline import load_or_create_windows
from models import HybridMAE, FusionHead
from training import (make_optimizer, make_plateau_scheduler,
                      train_autoencoder, random_mask)

CWRU_CHANNELS = {"de": [0], "fe": [1], "both": [0, 1]}
CWRU_FS = 48_000


@dataclass
class LoadData:
    X: np.ndarray
    y: np.ndarray
    fs: int
    classes: list[str]
    meta: dict


def load_windows(data_dir, load_id: int, *, window_size: int = 1024,
                 stride: int = 1024, channels: str = "both",
                 window_cache_dir=None, use_cache: bool = True) -> LoadData:
    X, y, meta = load_or_create_windows(
        data_dir, load_id, window_size=window_size, stride=stride,
        cache_dir=window_cache_dir, use_cache=use_cache,
    )
    X = np.asarray(X)[:, CWRU_CHANNELS[channels], :].astype(np.float32)
    y = np.asarray([str(v) for v in y])
    classes = sorted(np.unique(y).tolist())
    return LoadData(X=X, y=y, fs=CWRU_FS, classes=classes, meta=meta)


def make_split(y: np.ndarray, test_size: float,
               seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train/test split over windows."""
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []

    for label in np.unique(y):
        shuffled = rng.permutation(np.flatnonzero(y == label))
        n_test = max(1, int(round(len(shuffled) * test_size)))
        n_test = min(n_test, len(shuffled) - 1)
        test_idx.extend(shuffled[:n_test].tolist())
        train_idx.extend(shuffled[n_test:].tolist())

    return (rng.permutation(np.asarray(train_idx, dtype=int)),
            rng.permutation(np.asarray(test_idx, dtype=int)))


def stratified_val_split(y: np.ndarray, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for label in np.unique(y):
        pos = rng.permutation(np.flatnonzero(y == label))
        n_val = max(1, int(round(len(pos) * val_ratio)))
        n_val = min(n_val, len(pos) - 1)
        va.extend(pos[:n_val].tolist())
        tr.extend(pos[n_val:].tolist())
    return np.asarray(tr, dtype=int), np.asarray(va, dtype=int)


def labeled_subset(y: np.ndarray, spc: int, seed: int) -> np.ndarray:
    """Draw the few-shot budget of ``spc`` labelled windows per class."""
    rng = np.random.default_rng(seed)
    idx = []
    for label in np.unique(y):
        pos = rng.permutation(np.flatnonzero(y == label))
        idx.extend(pos[:spc].tolist())
    return np.asarray(idx, dtype=int)


def norm_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True) + 1e-8
    return mean, std


def apply_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(((X - mean) / std).astype(np.float32))


def apply_instance_norm(X: np.ndarray) -> torch.Tensor:
    """Per-window standardisation; removes the load-dependent amplitude offset."""
    m = X.mean(axis=2, keepdims=True)
    s = X.std(axis=2, keepdims=True) + 1e-8
    return torch.from_numpy(((X - m) / s).astype(np.float32))


def normalize(X: np.ndarray, mean, std, mode: str = "dataset") -> torch.Tensor:
    return apply_instance_norm(X) if mode == "instance" else apply_norm(X, mean, std)


def adapt_batchnorm(embedder, X_target: torch.Tensor, device: str = "cpu",
                    batch_size: int = 256):
    """AdaBN: re-estimate BatchNorm statistics on unlabelled target windows."""
    emb = copy.deepcopy(embedder).to(device)
    for m in emb.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None
            m.train()
    with torch.no_grad():
        for i in range(0, len(X_target), batch_size):
            emb(X_target[i:i + batch_size].to(device))
    emb.eval()
    return emb


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    sim = z @ z.t() / temperature
    n = z1.shape[0]
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


def augment_batch(x: torch.Tensor) -> torch.Tensor:
    """Amplitude scaling, additive noise and a circular time shift."""
    b, length = x.shape[0], x.shape[-1]
    x = x * torch.empty(b, 1, 1, device=x.device).uniform_(0.8, 1.2)
    std = x.std(dim=(1, 2), keepdim=True)
    x = x + 0.03 * std * torch.randn_like(x)
    shifts = torch.randint(0, length, (b,), device=x.device)
    idx = (torch.arange(length, device=x.device).unsqueeze(0) - shifts.unsqueeze(1)) % length
    return torch.gather(x, 2, idx.unsqueeze(1).expand(-1, x.shape[1], -1))


def pretrain_contrastive(ae: HybridMAE, X_domains: list[torch.Tensor],
                         val_tensor: torch.Tensor | None = None, *,
                         pretrain: str = "hybrid", mask_ratio: float = 0.35,
                         con_weight: float = 1.0, temperature: float = 0.2,
                         epochs: int = 120, batch_size: int = 64,
                         lr: float = 1e-3, weight_decay: float = 1e-4,
                         patience: int = 20, device: str = "cpu",
                         print_every: int = 10, log_prefix: str = "AE") -> dict:
    """Pretraining loop with domain-balanced batches and an NT-Xent term.

    Two augmented and masked views of every window are encoded; the projection
    head is used only here and discarded afterwards. With ``pretrain="hybrid"``
    the masked reconstruction loss is optimised alongside the contrastive one.
    """
    n_dom = len(X_domains)
    per_dom = max(2, batch_size // n_dom)
    steps = max(1, min(len(X) for X in X_domains) // per_dom)

    ae.to(device)
    proj = nn.Sequential(
        nn.Linear(ae.latent_channels, ae.latent_channels),
        nn.ReLU(inplace=True),
        nn.Linear(ae.latent_channels, 64),
    ).to(device)
    opt = make_optimizer(list(ae.parameters()) + list(proj.parameters()),
                         "adamw", lr, weight_decay)
    scheduler = make_plateau_scheduler(opt, mode="min", factor=0.5, patience=8)
    crit = nn.MSELoss()

    def _emb(z):
        return z.flatten(2).mean(-1)

    def _step_losses(x):
        v1, v2 = augment_batch(x), augment_batch(x)
        x1, x2 = random_mask(v1, mask_ratio), random_mask(v2, mask_ratio)
        if pretrain == "hybrid":
            recon, z1 = ae(x1)
            rec = crit(recon, v1)
        else:
            z1 = ae.encoder(x1)[0]
            rec = None
        z2 = ae.encoder(x2)[0]
        con = nt_xent(proj(_emb(z1)), proj(_emb(z2)), temperature)
        return rec, con

    has_val = val_tensor is not None and len(val_tensor) > 0
    best_loss, best_state, bad_epochs = float("inf"), None, 0

    for epoch in range(epochs):
        ae.train()
        proj.train()
        perms = [torch.randperm(len(X)) for X in X_domains]
        tot_rec, tot_con, tot = 0.0, 0.0, 0.0

        for s in range(steps):
            sl = [perm[s * per_dom:(s + 1) * per_dom] for perm in perms]
            x = torch.cat([X[idx].to(device) for X, idx in zip(X_domains, sl)], dim=0)
            rec, con = _step_losses(x)

            loss = con_weight * con
            if rec is not None:
                loss = loss + rec
                tot_rec += rec.detach().item()
            tot_con += con.detach().item()

            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.detach().item()

        train_rec, train_con, train_tot = tot_rec / steps, tot_con / steps, tot / steps

        if has_val:
            ae.eval()
            proj.eval()
            v_tot, n_batches = 0.0, 0
            with torch.no_grad():
                for i in range(0, len(val_tensor), batch_size):
                    bx = val_tensor[i:i + batch_size].to(device)
                    if len(bx) < 4:
                        continue
                    v_rec, v_con = _step_losses(bx)
                    v = con_weight * float(v_con)
                    if v_rec is not None:
                        v += float(v_rec)
                    v_tot += v
                    n_batches += 1
            val_loss = v_tot / max(1, n_batches)
        else:
            val_loss = train_tot

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = copy.deepcopy(ae.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if (epoch + 1) % print_every == 0:
            msg = f"  {log_prefix} [{epoch + 1}/{epochs}] "
            if pretrain == "hybrid":
                msg += f" rec={train_rec:.6f} "
            msg += f" con={train_con:.6f}  val={val_loss:.6f}  best={best_loss:.6f}"
            print(msg)

        if patience and bad_epochs >= patience:
            print(f"  {log_prefix} early stopping at epoch {epoch + 1} "
                  f"(best val={best_loss:.6f})")
            break

    if best_state is not None:
        ae.load_state_dict(best_state)
    return {"best_val": best_loss}


def pretrain_ae(in_channels: int, latent_channels: int,
                X_domains: list[torch.Tensor] | torch.Tensor,
                X_val: torch.Tensor, *, pretrain: str = "hybrid",
                con_weight: float = 1.0, temperature: float = 0.2,
                mask_ratio: float = 0.35,
                epochs: int = 120, batch_size: int = 64, lr: float = 1e-3,
                weight_decay: float = 1e-4, patience: int = 20,
                device: str = "cpu", log_prefix: str = "AE") -> HybridMAE:
    if isinstance(X_domains, torch.Tensor):
        X_domains = [X_domains]
    ae = HybridMAE(in_channels=in_channels, latent_channels=latent_channels)

    if pretrain == "mae":
        train_autoencoder(
            ae, torch.cat(X_domains, dim=0), val_tensor=X_val, epochs=epochs,
            batch_size=batch_size, lr=lr, weight_decay=weight_decay,
            patience=patience, mask_ratio=mask_ratio, device=device,
            print_every=max(1, epochs // 5), log_prefix=log_prefix,
        )
    else:
        pretrain_contrastive(
            ae, X_domains, val_tensor=X_val, pretrain=pretrain,
            mask_ratio=mask_ratio, con_weight=con_weight, temperature=temperature,
            epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
            patience=patience, device=device,
            print_every=max(1, epochs // 5), log_prefix=log_prefix,
        )
    ae.to(device).eval()
    return ae


def embed_batches(embedder, X: torch.Tensor, device: str = "cpu",
                  batch_size: int = 256) -> torch.Tensor:
    embedder.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            outs.append(embedder(X[i:i + batch_size].to(device)).detach())
    return torch.cat(outs, dim=0)


def fit_head_on_embeddings(E_lab: torch.Tensor, y_lab: torch.Tensor,
                           E_val: torch.Tensor | None, y_val: torch.Tensor | None,
                           num_classes: int, num_experts: int, latent_channels: int,
                           *, hidden: int = 128, dropout: float = 0.3,
                           epochs: int = 120, lr: float = 1e-3,
                           weight_decay: float = 1e-4, patience: int = 20,
                           device: str = "cpu", monitor: str = "macro_f1") -> FusionHead:
    mlp = FusionHead(num_experts, latent_channels, num_classes,
                     hidden=hidden, dropout=dropout).to(device)
    opt = make_optimizer(mlp.parameters(), "adamw", lr, weight_decay)
    sched = make_plateau_scheduler(opt, mode="max", factor=0.5, patience=5, enabled=True)
    ce = nn.CrossEntropyLoss()

    y_lab_d = y_lab.to(device)
    best_score, best_state, no_improve = -np.inf, None, 0

    for epoch in range(epochs):
        mlp.train()
        opt.zero_grad()
        loss = ce(mlp(E_lab), y_lab_d)
        loss.backward()
        opt.step()

        mlp.eval()
        with torch.no_grad():
            if E_val is not None:
                preds = mlp(E_val).argmax(1).cpu().numpy()
                tgt = y_val.cpu().numpy()
            else:
                preds = mlp(E_lab).argmax(1).cpu().numpy()
                tgt = y_lab.cpu().numpy()
        score = (f1_score(tgt, preds, average="macro", zero_division=0)
                 if monitor == "macro_f1" else accuracy_score(tgt, preds))
        if sched is not None:
            sched.step(score)
        if score > best_score + 1e-4:
            best_score, best_state, no_improve = score, copy.deepcopy(mlp.state_dict()), 0
        else:
            no_improve += 1
            if patience and no_improve >= patience:
                break

    if best_state is not None:
        mlp.load_state_dict(best_state)
    return mlp


def train_fusion_head(embedder, X_lab: torch.Tensor, y_lab: torch.Tensor,
                      X_val: torch.Tensor | None, y_val: torch.Tensor | None,
                      num_classes: int, *, hidden: int = 128, dropout: float = 0.3,
                      epochs: int = 120, lr: float = 1e-3,
                      weight_decay: float = 1e-4, patience: int = 20,
                      device: str = "cpu", monitor: str = "macro_f1") -> FusionHead:
    E_lab = embed_batches(embedder, X_lab, device)
    E_val = (embed_batches(embedder, X_val, device)
             if X_val is not None and len(X_val) else None)
    return fit_head_on_embeddings(
        E_lab, y_lab, E_val, y_val, num_classes,
        embedder.num_experts, embedder.latent_channels,
        hidden=hidden, dropout=dropout, epochs=epochs, lr=lr,
        weight_decay=weight_decay, patience=patience, device=device, monitor=monitor,
    )


def predict(embedder, head, X: torch.Tensor, device: str = "cpu",
            batch_size: int = 256) -> np.ndarray:
    embedder.eval()
    head.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            logits = head(embedder(X[i:i + batch_size].to(device)))
            preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds) if preds else np.empty(0, dtype=int)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

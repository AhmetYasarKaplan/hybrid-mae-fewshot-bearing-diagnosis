from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset


def make_optimizer(parameters, optimizer_name="adamw", lr=1e-3, weight_decay=1e-4):
    name = optimizer_name.lower()
    if name == "adam":
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}. Use 'adam' or 'adamw'.")


def make_plateau_scheduler(optimizer, mode="min", factor=0.5, patience=5,
                           min_lr=1e-6, enabled=True):
    if not enabled:
        return None
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=mode, factor=factor, patience=patience, min_lr=min_lr)


def _current_lr(optimizer):
    return float(optimizer.param_groups[0]["lr"])


def _state_dict_copy(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _load_state_dict_copy(model, state):
    model.load_state_dict({k: v.clone() for k, v in state.items()})


def random_mask(x: torch.Tensor, mask_ratio: float = 0.35) -> torch.Tensor:
    """Mask a fraction of each window: random points, several blocks, or one span."""
    B, _, L = x.shape
    mask = torch.ones_like(x)

    for i in range(B):
        strategy = torch.randint(0, 3, (1,)).item()

        if strategy == 0:
            point_mask = (torch.rand(L, device=x.device) > mask_ratio).float()
            mask[i, :, :] = point_mask.unsqueeze(0)
        elif strategy == 1:
            num_blocks = torch.randint(3, 8, (1,)).item()
            per_block = int(L * mask_ratio) // num_blocks
            for _ in range(num_blocks):
                start = torch.randint(0, max(1, L - per_block), (1,)).item()
                mask[i, :, start:start + per_block] = 0
        else:
            num_mask = int(L * mask_ratio)
            start = torch.randint(0, max(1, L - num_mask), (1,)).item()
            mask[i, :, start:start + num_mask] = 0

    return x * mask


def evaluate_ae_loss(autoencoder, loader, criterion, device="cpu", mask_ratio=None):
    autoencoder.eval()
    total = 0.0
    with torch.no_grad():
        for bx, _ in loader:
            bx = bx.to(device)
            bx_in = random_mask(bx, mask_ratio) if mask_ratio else bx
            x_recon, _ = autoencoder(bx_in)
            total += criterion(x_recon, bx).item()
    return total / max(1, len(loader))


def evaluate_classifier(model, X_tensor, y_tensor, criterion, device="cpu",
                        batch_size=128):
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size,
                        shuffle=False)
    losses, preds, targets = [], [], []

    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            losses.append(criterion(logits, by).item())
            preds.extend(logits.argmax(1).cpu().numpy().tolist())
            targets.extend(by.cpu().numpy().tolist())

    if not targets:
        return {"loss": 0.0, "accuracy": 0.0, "macro_f1": 0.0}
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
    }


def train_autoencoder(autoencoder, X_tensor, val_tensor=None, epochs=120,
                      batch_size=32, lr=1e-3, weight_decay=1e-4,
                      optimizer_name="adamw", patience=20, min_delta=1e-5,
                      mask_ratio=None, device="cpu", print_every=10,
                      log_prefix="AE"):
    """Masked-reconstruction pretraining with early stopping on the validation loss."""
    train_loader = DataLoader(TensorDataset(X_tensor, X_tensor),
                              batch_size=batch_size, shuffle=True)
    val_loader = None
    if val_tensor is not None and len(val_tensor) > 0:
        val_loader = DataLoader(TensorDataset(val_tensor, val_tensor),
                                batch_size=batch_size, shuffle=False)

    opt = make_optimizer(autoencoder.parameters(), optimizer_name, lr, weight_decay)
    scheduler = make_plateau_scheduler(opt, mode="min", factor=0.5, patience=8)
    crit = nn.MSELoss()

    best_loss, best_state, best_epoch, no_improve = float("inf"), None, None, 0
    autoencoder.to(device)

    for epoch in range(epochs):
        autoencoder.train()
        total = 0.0
        for bx, _ in train_loader:
            bx = bx.to(device)
            bx_in = random_mask(bx, mask_ratio) if mask_ratio else bx
            opt.zero_grad()
            x_recon, _ = autoencoder(bx_in)
            loss = crit(x_recon, bx)
            loss.backward()
            opt.step()
            total += loss.item()

        train_loss = total / max(1, len(train_loader))
        val_loss = (evaluate_ae_loss(autoencoder, val_loader, crit, device, mask_ratio)
                    if val_loader is not None else train_loss)

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_loss - min_delta:
            best_loss, best_state, best_epoch, no_improve = (
                val_loss, _state_dict_copy(autoencoder), epoch + 1, 0)
        else:
            no_improve += 1

        if (epoch + 1) % print_every == 0:
            print(f"  {log_prefix} [{epoch + 1}/{epochs}]  "
                  f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                  f"best_val={best_loss:.6f}  lr={_current_lr(opt):.2e}")

        if patience and no_improve >= patience:
            print(f"  {log_prefix} early stopping at epoch {epoch + 1}; "
                  f"best epoch={best_epoch} val_loss={best_loss:.6f}")
            break

    if best_state is not None:
        _load_state_dict_copy(autoencoder, best_state)
    return {"best_epoch": best_epoch, "best_val": float(best_loss)}


def train_cnn(model, labeled_loader, X_labeled, y_labeled, X_val=None, y_val=None,
              epochs=80, lr=1e-3, weight_decay=1e-4, optimizer_name="adamw",
              patience=15, min_delta=1e-4, monitor="macro_f1", device="cpu",
              print_every=10):
    """Supervised training loop for the CNN1D baseline."""
    model.to(device)
    opt = make_optimizer(model.parameters(), optimizer_name, lr, weight_decay)
    scheduler = make_plateau_scheduler(opt, mode="max", factor=0.5, patience=5)
    crit = nn.CrossEntropyLoss()

    best_score, best_state, best_epoch, no_improve = -float("inf"), None, None, 0

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for bx, by in labeled_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
            total += loss.item()
        train_loss = total / max(1, len(labeled_loader))

        model.eval()
        if X_val is not None and y_val is not None and len(X_val) > 0:
            metrics = evaluate_classifier(model, X_val, y_val, crit, device=device)
        else:
            metrics = evaluate_classifier(model, X_labeled, y_labeled, crit, device=device)

        current = metrics[monitor]
        if scheduler is not None:
            scheduler.step(current)

        if current > best_score + min_delta:
            best_score, best_state, best_epoch, no_improve = (
                current, _state_dict_copy(model), epoch + 1, 0)
        else:
            no_improve += 1

        if (epoch + 1) % print_every == 0:
            print(f"  CNN [{epoch + 1}/{epochs}]  train_loss={train_loss:.4f}  "
                  f"val_loss={metrics['loss']:.4f}  val_acc={metrics['accuracy']:.4f}  "
                  f"val_f1={metrics['macro_f1']:.4f}  lr={_current_lr(opt):.2e}")

        if patience and no_improve >= patience:
            print(f"  CNN early stopping at epoch {epoch + 1}; "
                  f"best epoch={best_epoch} {monitor}={best_score:.4f}")
            break

    if best_state is not None:
        _load_state_dict_copy(model, best_state)
    return {"best_epoch": best_epoch, "best_metric": float(best_score)}


def get_cnn_predictions(model, X_tensor, device="cpu", batch_size=256):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            preds.append(model(X_tensor[i:i + batch_size].to(device))
                         .argmax(1).cpu().numpy())
    return np.concatenate(preds) if preds else np.empty(0, dtype=int)

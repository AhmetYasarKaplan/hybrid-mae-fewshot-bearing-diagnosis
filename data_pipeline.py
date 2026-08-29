from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_WINDOW_SIZE = 1024
DEFAULT_STRIDE = 1024


def _normalise_label_array(labels: Iterable[str]) -> np.ndarray:
    return np.asarray([str(label) for label in labels])


def ensure_channel_first(windows: np.ndarray) -> np.ndarray:
    if windows.ndim == 2:
        return windows[:, None, :]
    if windows.ndim != 3:
        raise ValueError(f"Expected windows with 2 or 3 dimensions, got {windows.shape}")
    return windows


def create_windows_from_csv(
    csv_path: Path,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    signal_columns: tuple[str, ...] = ("DE_data", "FE_data"),
    fault_column: str = "fault",
) -> tuple[np.ndarray, np.ndarray, dict]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {csv_path}")

    by_fault: dict[str, list[list[float]]] = defaultdict(
        lambda: [[] for _ in signal_columns]
    )

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = {fault_column, *signal_columns} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

        for row in reader:
            fault = row[fault_column]
            channels = by_fault[fault]
            for idx, column in enumerate(signal_columns):
                channels[idx].append(float(row[column]))

    all_windows: list[np.ndarray] = []
    all_labels: list[str] = []
    class_counts: dict[str, dict[str, int]] = {}

    for fault in sorted(by_fault):
        channel_arrays = [np.asarray(values, dtype=np.float64) for values in by_fault[fault]]
        n_samples = len(channel_arrays[0])
        n_windows = (n_samples - window_size) // stride + 1
        if n_windows <= 0:
            class_counts[fault] = {"samples": n_samples, "windows": 0}
            continue

        fault_windows = np.empty(
            (n_windows, len(signal_columns), window_size),
            dtype=np.float64,
        )
        for window_idx in range(n_windows):
            start = window_idx * stride
            stop = start + window_size
            for channel_idx, values in enumerate(channel_arrays):
                fault_windows[window_idx, channel_idx, :] = values[start:stop]

        all_windows.append(fault_windows)
        all_labels.extend([fault] * n_windows)
        class_counts[fault] = {"samples": n_samples, "windows": int(n_windows)}

    if not all_windows:
        raise ValueError(f"No windows could be created from {csv_path}")

    windows = np.concatenate(all_windows, axis=0)
    labels = _normalise_label_array(all_labels)
    metadata = {
        "source_csv": str(csv_path),
        "window_size": int(window_size),
        "stride": int(stride),
        "signal_columns": list(signal_columns),
        "fault_column": fault_column,
        "class_counts": class_counts,
        "total_windows": int(len(labels)),
    }
    return windows, labels, metadata


def window_cache_path(
    data_dir: Path,
    load_id: int,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    cache_dir: Path | None = None,
) -> Path:
    cache_root = Path(cache_dir) if cache_dir is not None else Path(data_dir) / "generated_windows"
    return cache_root / f"load_{load_id}_all_faults_windows_ws{window_size}_stride{stride}.npz"


def load_or_create_windows(
    data_dir: Path,
    load_id: int,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    data_dir = Path(data_dir)
    cache_path = window_cache_path(data_dir, load_id, window_size, stride, cache_dir)
    metadata_path = cache_path.with_suffix(".json")

    if use_cache and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update({"cache_path": str(cache_path), "loaded_from_cache": True})
        return ensure_channel_first(cached["windows"]), _normalise_label_array(cached["labels"]), metadata

    csv_path = data_dir / f"load_{load_id}_all_faults.csv"
    windows, labels, metadata = create_windows_from_csv(csv_path, window_size, stride)
    metadata.update({"cache_path": str(cache_path), "loaded_from_cache": False})

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, windows=windows, labels=labels)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return windows, labels, metadata

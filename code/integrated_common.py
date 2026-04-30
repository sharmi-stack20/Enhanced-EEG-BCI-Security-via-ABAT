from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE2_PREPROCESSED_ROOT = PROJECT_ROOT / "results" / "phase2_preprocessing" / "preprocessed"
PHASE4_ROOT = PROJECT_ROOT / "results" / "phase4_model_training"
PHASE5_ROOT = PROJECT_ROOT / "results" / "phase5_abat_automation"
PHASE6_ROOT = PROJECT_ROOT / "results" / "phase6_reporting_visualization"

SAMPLING_RATES = {"public": 250.0, "lab": 512.0}
WINDOW_SECONDS = 2.0
STRIDE_SECONDS = 1.0
CHANNELS_TO_USE = 8
TIME_BINS = 64
RANDOM_STATE = 42


@dataclass
class RecordingMeta:
    dataset: str
    record_name: str
    label: str
    file_path: Path
    sampling_rate_hz: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_lab_label(stem: str) -> str:
    if "(" in stem and ")" in stem:
        return stem.split("(")[-1].split(")")[0]
    return "unknown"


def list_recordings(dataset: str) -> list[RecordingMeta]:
    if dataset == "public":
        records: list[RecordingMeta] = []
        for subject_dir in sorted((PHASE2_PREPROCESSED_ROOT / "public").glob("*")):
            if not subject_dir.is_dir():
                continue
            label = subject_dir.name[-1]
            for run_file in sorted(subject_dir.glob("run_*.npy")):
                records.append(
                    RecordingMeta(
                        dataset="public",
                        record_name=f"{subject_dir.name}_{run_file.stem}",
                        label=label,
                        file_path=run_file,
                        sampling_rate_hz=SAMPLING_RATES["public"],
                    )
                )
        return records

    if dataset == "lab":
        records = []
        for npy_file in sorted((PHASE2_PREPROCESSED_ROOT / "lab").glob("*.npy")):
            label = parse_lab_label(npy_file.stem)
            if label == "unknown":
                continue
            records.append(
                RecordingMeta(
                    dataset="lab",
                    record_name=npy_file.stem,
                    label=label,
                    file_path=npy_file,
                    sampling_rate_hz=SAMPLING_RATES["lab"],
                )
            )
        return records

    raise ValueError(f"Unsupported dataset: {dataset}")


def make_windows(signal: np.ndarray, sampling_rate_hz: float, window_seconds: float = WINDOW_SECONDS, stride_seconds: float = STRIDE_SECONDS) -> np.ndarray:
    window_samples = int(round(sampling_rate_hz * window_seconds))
    stride_samples = int(round(sampling_rate_hz * stride_seconds))
    if signal.shape[0] < window_samples:
        return np.empty((0, window_samples, signal.shape[1]), dtype=np.float32)

    windows = []
    for start in range(0, signal.shape[0] - window_samples + 1, stride_samples):
        stop = start + window_samples
        windows.append(signal[start:stop].astype(np.float32))
    return np.stack(windows, axis=0) if windows else np.empty((0, window_samples, signal.shape[1]), dtype=np.float32)


def compact_window(window: np.ndarray, channels_to_use: int = CHANNELS_TO_USE, time_bins: int = TIME_BINS) -> np.ndarray:
    usable = window[:, : min(channels_to_use, window.shape[1])].astype(np.float32)
    usable = usable - usable.mean(axis=0, keepdims=True)
    std = usable.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    usable = usable / std

    boundaries = np.linspace(0, usable.shape[0], time_bins + 1, dtype=int)
    pooled = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        segment = usable[start:end]
        pooled.append(segment.mean(axis=0) if len(segment) else np.zeros(usable.shape[1], dtype=np.float32))
    compact = np.stack(pooled, axis=0).T.reshape(-1)
    return np.nan_to_num(compact, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)


def build_window_dataset(dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    records = list_recordings(dataset)
    label_names = sorted({record.label for record in records})
    label_to_id = {label: idx for idx, label in enumerate(label_names)}

    X_list, y_list, groups_list, summary_rows = [], [], [], []
    for record in records:
        signal = np.load(record.file_path)
        windows = make_windows(signal, record.sampling_rate_hz)
        if len(windows) == 0:
            continue
        compact = np.stack([compact_window(window) for window in windows]).astype(np.float32)
        X_list.append(compact)
        y_list.append(np.full(len(compact), label_to_id[record.label], dtype=np.int64))
        groups_list.append(np.array([record.record_name] * len(compact)))
        summary_rows.append(
            {
                "record_name": record.record_name,
                "label": record.label,
                "windows": int(len(compact)),
                "feature_dim": int(compact.shape[1]),
            }
        )

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    groups = np.concatenate(groups_list)
    return X, y, groups, label_names, pd.DataFrame(summary_rows)


def stratified_group_split(groups: np.ndarray, labels: np.ndarray, test_fraction: float = 0.25, random_state: int = RANDOM_STATE) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    unique_groups, first_indices = np.unique(groups, return_index=True)
    group_labels = labels[first_indices]
    train_groups, test_groups = [], []

    for label in np.unique(group_labels):
        label_groups = unique_groups[group_labels == label]
        shuffled = rng.permutation(label_groups)
        n_test = max(1, int(round(len(label_groups) * test_fraction)))
        n_test = min(n_test, max(1, len(label_groups) - 1)) if len(label_groups) > 1 else 1
        test_groups.extend(shuffled[:n_test].tolist())
        train_groups.extend(shuffled[n_test:].tolist())

    train_mask = np.isin(groups, np.array(train_groups, dtype=groups.dtype))
    test_mask = np.isin(groups, np.array(test_groups, dtype=groups.dtype))
    return train_mask, test_mask


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> tuple[dict, pd.DataFrame, np.ndarray]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    report_df = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(label_names)),
            target_names=label_names,
            zero_division=0,
            output_dict=True,
        )
    ).transpose()
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_names)))
    return metrics, report_df, cm

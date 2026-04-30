from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis, skew

from phase1_data_preparation import extract_condition

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE2_DIR = PROJECT_ROOT / "results" / "phase2_preprocessing" / "preprocessed"
PHASE3_RESULTS_DIR = PROJECT_ROOT / "results" / "phase3_feature_extraction"
FEATURES_DIR = PHASE3_RESULTS_DIR / "features"
PLOTS_DIR = PHASE3_RESULTS_DIR / "plots"

WINDOW_SECONDS = {"public": 2.0, "lab": 2.0}
SAMPLING_RATES = {"public": 250.0, "lab": 512.0}
CHANNELS_TO_USE = 8


def ensure_directories() -> None:
    for path in [
        PHASE3_RESULTS_DIR,
        FEATURES_DIR / "public",
        FEATURES_DIR / "lab",
        PLOTS_DIR / "public",
        PLOTS_DIR / "lab",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def split_into_windows(eeg: np.ndarray, sampling_rate_hz: float, window_seconds: float) -> np.ndarray:
    samples_per_window = max(1, int(round(sampling_rate_hz * window_seconds)))
    full_windows = eeg.shape[0] // samples_per_window
    if full_windows == 0:
        raise ValueError("Recording is too short for one full feature window.")
    trimmed = eeg[: full_windows * samples_per_window]
    return trimmed.reshape(full_windows, samples_per_window, eeg.shape[1])


def bandpower(signal_1d: np.ndarray, sampling_rate_hz: float, low_hz: float, high_hz: float) -> float:
    frequencies, psd = welch(signal_1d, fs=sampling_rate_hz, nperseg=min(256, len(signal_1d)))
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], frequencies[mask]))


def extract_window_features(window: np.ndarray, sampling_rate_hz: float) -> dict:
    usable = window[:, : min(CHANNELS_TO_USE, window.shape[1])]
    mean_signal = usable.mean(axis=1)

    feature_row: dict[str, float] = {}

    for channel_idx in range(usable.shape[1]):
        channel = usable[:, channel_idx]
        prefix = f"ch{channel_idx + 1}"
        feature_row[f"{prefix}_mean"] = float(np.mean(channel))
        feature_row[f"{prefix}_std"] = float(np.std(channel))
        feature_row[f"{prefix}_rms"] = float(np.sqrt(np.mean(np.square(channel))))

    feature_row["global_mean_abs"] = float(np.mean(np.abs(usable)))
    feature_row["global_skewness"] = float(skew(mean_signal, bias=False))
    feature_row["global_kurtosis"] = float(kurtosis(mean_signal, fisher=True, bias=False))
    feature_row["line_length"] = float(np.sum(np.abs(np.diff(mean_signal))))
    feature_row["signal_energy"] = float(np.mean(np.square(mean_signal)))
    feature_row["theta_power"] = bandpower(mean_signal, sampling_rate_hz, 4.0, 8.0)
    feature_row["alpha_power"] = bandpower(mean_signal, sampling_rate_hz, 8.0, 13.0)
    feature_row["beta_power"] = bandpower(mean_signal, sampling_rate_hz, 13.0, 30.0)
    return feature_row


def save_preview_plot(feature_df: pd.DataFrame, output_path: Path, title: str) -> None:
    columns = [col for col in ["global_mean_abs", "signal_energy", "alpha_power", "beta_power"] if col in feature_df.columns]
    if not columns:
        return

    fig, axes = plt.subplots(len(columns), 1, figsize=(12, 2.8 * len(columns)), sharex=True)
    if len(columns) == 1:
        axes = [axes]

    x = np.arange(1, len(feature_df) + 1)
    for axis, column in zip(axes, columns):
        axis.plot(x, feature_df[column], marker="o", linewidth=1)
        axis.set_ylabel(column)
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Window Index")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def process_public_dataset() -> pd.DataFrame:
    summary_rows = []

    for subject_dir in sorted((PHASE2_DIR / "public").glob("*")):
        if not subject_dir.is_dir():
            continue

        output_subject_dir = FEATURES_DIR / "public" / subject_dir.name
        output_subject_dir.mkdir(parents=True, exist_ok=True)

        for run_file in sorted(subject_dir.glob("run_*.npy")):
            eeg = np.load(run_file)
            windows = split_into_windows(eeg, SAMPLING_RATES["public"], WINDOW_SECONDS["public"])
            feature_rows = [extract_window_features(window, SAMPLING_RATES["public"]) for window in windows]
            feature_df = pd.DataFrame(feature_rows)
            feature_df.insert(0, "window_index", np.arange(1, len(feature_df) + 1))

            output_csv = output_subject_dir / f"{run_file.stem}_features.csv"
            feature_df.to_csv(output_csv, index=False)

            if run_file.stem == "run_01":
                save_preview_plot(
                    feature_df,
                    PLOTS_DIR / "public" / f"{subject_dir.name}_{run_file.stem}.png",
                    title=f"Phase 3 | Public | {subject_dir.name} | {run_file.stem}",
                )

            summary_rows.append(
                {
                    "dataset": "public",
                    "subject": subject_dir.name,
                    "run": run_file.stem,
                    "windows": int(len(feature_df)),
                    "features_per_window": int(feature_df.shape[1] - 1),
                    "mean_signal_energy": float(feature_df["signal_energy"].mean()),
                    "mean_alpha_power": float(feature_df["alpha_power"].mean()),
                    "mean_beta_power": float(feature_df["beta_power"].mean()),
                }
            )

    return pd.DataFrame(summary_rows)


def process_lab_dataset() -> pd.DataFrame:
    summary_rows = []

    for file_path in sorted((PHASE2_DIR / "lab").glob("*.npy")):
        eeg = np.load(file_path)
        windows = split_into_windows(eeg, SAMPLING_RATES["lab"], WINDOW_SECONDS["lab"])
        feature_rows = [extract_window_features(window, SAMPLING_RATES["lab"]) for window in windows]
        feature_df = pd.DataFrame(feature_rows)
        feature_df.insert(0, "window_index", np.arange(1, len(feature_df) + 1))

        output_csv = FEATURES_DIR / "lab" / f"{file_path.stem}_features.csv"
        feature_df.to_csv(output_csv, index=False)

        save_preview_plot(
            feature_df,
            PLOTS_DIR / "lab" / f"{file_path.stem}.png",
            title=f"Phase 3 | Lab | {file_path.stem}",
        )

        summary_rows.append(
            {
                "dataset": "lab",
                "file_name": file_path.stem,
                "condition": extract_condition(file_path.stem),
                "windows": int(len(feature_df)),
                "features_per_window": int(feature_df.shape[1] - 1),
                "mean_signal_energy": float(feature_df["signal_energy"].mean()),
                "mean_alpha_power": float(feature_df["alpha_power"].mean()),
                "mean_beta_power": float(feature_df["beta_power"].mean()),
            }
        )

    return pd.DataFrame(summary_rows)


def build_combined_summary(public_df: pd.DataFrame, lab_df: pd.DataFrame) -> pd.DataFrame:
    public_view = public_df.rename(columns={"subject": "record_group", "run": "record_name"})
    lab_view = lab_df.rename(columns={"condition": "record_group", "file_name": "record_name"})
    return pd.concat([public_view, lab_view], ignore_index=True, sort=False)


def main() -> None:
    ensure_directories()

    public_df = process_public_dataset()
    lab_df = process_lab_dataset()
    combined_df = build_combined_summary(public_df, lab_df)

    public_df.to_csv(PHASE3_RESULTS_DIR / "public_feature_summary.csv", index=False)
    lab_df.to_csv(PHASE3_RESULTS_DIR / "lab_feature_summary.csv", index=False)
    combined_df.to_csv(PHASE3_RESULTS_DIR / "combined_feature_summary.csv", index=False)

    print("Phase 3 complete.")
    print(f"Public feature files: {len(public_df)}")
    print(f"Lab feature files: {len(lab_df)}")
    print(f"Results saved under: {PHASE3_RESULTS_DIR}")


if __name__ == "__main__":
    main()

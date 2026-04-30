from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, filtfilt, iirnotch

from phase1_data_preparation import extract_condition, infer_sampling_rate, unwrap_public_run

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DATASET_DIR = Path("/Users/saisharmilakalva/Desktop/Public_Dataset")
LAB_DATASET_DIR = Path("/Users/saisharmilakalva/Desktop/Lab_Dataset")
PHASE2_RESULTS_DIR = PROJECT_ROOT / "results" / "phase2_preprocessing"
PREPROCESSED_DIR = PHASE2_RESULTS_DIR / "preprocessed"
PLOTS_DIR = PHASE2_RESULTS_DIR / "plots"

BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 50.0
LAB_NOTCH_HZ = 50.0
LAB_NOTCH_Q = 30.0


def ensure_directories() -> None:
    for path in [
        PHASE2_RESULTS_DIR,
        PREPROCESSED_DIR / "public",
        PREPROCESSED_DIR / "lab",
        PLOTS_DIR / "public",
        PLOTS_DIR / "lab",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def bandpass_filter(eeg: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
    nyquist = 0.5 * sampling_rate_hz
    low = BANDPASS_LOW_HZ / nyquist
    high = min(BANDPASS_HIGH_HZ / nyquist, 0.99)
    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, eeg, axis=0)


def notch_filter(eeg: np.ndarray, sampling_rate_hz: float, notch_hz: float = LAB_NOTCH_HZ) -> np.ndarray:
    nyquist = 0.5 * sampling_rate_hz
    normalized_notch = notch_hz / nyquist
    if normalized_notch >= 1:
        return eeg
    b, a = iirnotch(normalized_notch, LAB_NOTCH_Q)
    return filtfilt(b, a, eeg, axis=0)


def preprocess_public_signal(eeg: np.ndarray) -> np.ndarray:
    return bandpass_filter(eeg, sampling_rate_hz=250.0).astype(np.float32)


def preprocess_lab_signal(eeg: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
    filtered = bandpass_filter(eeg, sampling_rate_hz=sampling_rate_hz)
    filtered = notch_filter(filtered, sampling_rate_hz=sampling_rate_hz)
    return filtered.astype(np.float32)


def save_preview_plot(
    raw_eeg: np.ndarray,
    filtered_eeg: np.ndarray,
    output_path: Path,
    title: str,
    max_channels: int = 2,
    max_samples: int = 2000,
) -> None:
    channels_to_plot = min(max_channels, raw_eeg.shape[1], filtered_eeg.shape[1])
    sample_limit = min(max_samples, raw_eeg.shape[0], filtered_eeg.shape[0])
    time_axis = np.arange(sample_limit)

    fig, axes = plt.subplots(channels_to_plot, 2, figsize=(14, 3.2 * channels_to_plot), sharex=True)
    if channels_to_plot == 1:
        axes = np.array([axes])

    for channel_idx in range(channels_to_plot):
        axes[channel_idx, 0].plot(time_axis, raw_eeg[:sample_limit, channel_idx], linewidth=0.8)
        axes[channel_idx, 0].set_title(f"Raw Ch {channel_idx + 1}")
        axes[channel_idx, 0].grid(True, alpha=0.3)

        axes[channel_idx, 1].plot(time_axis, filtered_eeg[:sample_limit, channel_idx], linewidth=0.8)
        axes[channel_idx, 1].set_title(f"Filtered Ch {channel_idx + 1}")
        axes[channel_idx, 1].grid(True, alpha=0.3)

    for axis in axes[-1]:
        axis.set_xlabel("Samples")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def summarize_filtered_signal(
    dataset: str,
    file_name: str,
    record_id: str,
    filtered_eeg: np.ndarray,
    sampling_rate_hz: float,
    **extra_fields,
) -> dict:
    return {
        "dataset": dataset,
        "file_name": file_name,
        "record_id": record_id,
        "samples": int(filtered_eeg.shape[0]),
        "channels": int(filtered_eeg.shape[1]),
        "sampling_rate_hz": float(sampling_rate_hz),
        "mean_amplitude": float(np.mean(filtered_eeg)),
        "std_amplitude": float(np.std(filtered_eeg)),
        "min_amplitude": float(np.min(filtered_eeg)),
        "max_amplitude": float(np.max(filtered_eeg)),
        **extra_fields,
    }


def process_public_dataset() -> pd.DataFrame:
    summary_rows = []

    for file_path in sorted(PUBLIC_DATASET_DIR.glob("*.mat")):
        subject_dir = PREPROCESSED_DIR / "public" / file_path.stem
        subject_dir.mkdir(parents=True, exist_ok=True)

        mat_data = sio.loadmat(file_path)
        data_cell = mat_data["data"]

        for run_idx in range(data_cell.shape[1]):
            raw_eeg = unwrap_public_run(data_cell[0, run_idx])
            filtered_eeg = preprocess_public_signal(raw_eeg)

            run_name = f"run_{run_idx + 1:02d}"
            np.save(subject_dir / f"{run_name}.npy", filtered_eeg)

            if run_idx == 0:
                save_preview_plot(
                    raw_eeg,
                    filtered_eeg,
                    PLOTS_DIR / "public" / f"{file_path.stem}_{run_name}.png",
                    title=f"Phase 2 | Public | {file_path.stem} | {run_name}",
                )

            summary_rows.append(
                summarize_filtered_signal(
                    dataset="public",
                    file_name=file_path.name,
                    record_id=f"{file_path.stem}_{run_name}",
                    filtered_eeg=filtered_eeg,
                    sampling_rate_hz=250.0,
                    subject=file_path.stem,
                    run=run_idx + 1,
                )
            )

    return pd.DataFrame(summary_rows)


def process_lab_dataset() -> pd.DataFrame:
    summary_rows = []

    for file_path in sorted(LAB_DATASET_DIR.glob("*.mat")):
        with h5py.File(file_path, "r") as mat_file:
            if "y" not in mat_file:
                continue
            matrix = np.array(mat_file["y"])

        time_values = matrix[:, 0]
        raw_eeg = matrix[:, 1:]
        sampling_rate_hz = infer_sampling_rate(time_values)
        if sampling_rate_hz is None:
            continue

        filtered_eeg = preprocess_lab_signal(raw_eeg, sampling_rate_hz)
        np.save(PREPROCESSED_DIR / "lab" / f"{file_path.stem}.npy", filtered_eeg)

        save_preview_plot(
            raw_eeg,
            filtered_eeg,
            PLOTS_DIR / "lab" / f"{file_path.stem}.png",
            title=f"Phase 2 | Lab | {file_path.stem}",
        )

        summary_rows.append(
            summarize_filtered_signal(
                dataset="lab",
                file_name=file_path.name,
                record_id=file_path.stem,
                filtered_eeg=filtered_eeg,
                sampling_rate_hz=sampling_rate_hz,
                condition=extract_condition(file_path.stem),
                duration_seconds=float(time_values[-1] - time_values[0]) if len(time_values) > 1 else 0.0,
            )
        )

    return pd.DataFrame(summary_rows)


def build_combined_summary(public_df: pd.DataFrame, lab_df: pd.DataFrame) -> pd.DataFrame:
    public_view = public_df[
        [
            "dataset",
            "file_name",
            "record_id",
            "samples",
            "channels",
            "sampling_rate_hz",
            "mean_amplitude",
            "std_amplitude",
        ]
    ]
    lab_view = lab_df[
        [
            "dataset",
            "file_name",
            "record_id",
            "samples",
            "channels",
            "sampling_rate_hz",
            "mean_amplitude",
            "std_amplitude",
        ]
    ]
    return pd.concat([public_view, lab_view], ignore_index=True, sort=False)


def main() -> None:
    ensure_directories()

    public_df = process_public_dataset()
    lab_df = process_lab_dataset()
    combined_df = build_combined_summary(public_df, lab_df)

    public_df.to_csv(PHASE2_RESULTS_DIR / "public_preprocessing_summary.csv", index=False)
    lab_df.to_csv(PHASE2_RESULTS_DIR / "lab_preprocessing_summary.csv", index=False)
    combined_df.to_csv(PHASE2_RESULTS_DIR / "combined_preprocessing_summary.csv", index=False)

    print("Phase 2 complete.")
    print(f"Public preprocessed runs: {len(public_df)}")
    print(f"Lab preprocessed records: {len(lab_df)}")
    print(f"Results saved under: {PHASE2_RESULTS_DIR}")


if __name__ == "__main__":
    main()

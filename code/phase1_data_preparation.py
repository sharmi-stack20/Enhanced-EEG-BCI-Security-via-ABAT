from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io as sio

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DATASET_DIR = Path("/Users/saisharmilakalva/Desktop/Public_Dataset")
LAB_DATASET_DIR = Path("/Users/saisharmilakalva/Desktop/Lab_Dataset")
PHASE1_RESULTS_DIR = PROJECT_ROOT / "results" / "phase1_data_preparation"
PLOTS_DIR = PHASE1_RESULTS_DIR / "plots"


def ensure_directories() -> None:
    PHASE1_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def infer_sampling_rate(time_values: np.ndarray) -> float | None:
    deltas = np.diff(time_values)
    deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    if deltas.size == 0:
        return None
    return float(1.0 / np.median(deltas))


def unwrap_public_run(run_struct) -> np.ndarray:
    eeg = run_struct["X"]

    while isinstance(eeg, np.ndarray) and eeg.size == 1:
        eeg = eeg[0]

    eeg = np.array(eeg)
    if eeg.ndim == 3:
        eeg = eeg[0]
    return eeg


def summarize_public_dataset() -> pd.DataFrame:
    summary_rows = []

    for file_path in sorted(PUBLIC_DATASET_DIR.glob("*.mat")):
        mat_data = sio.loadmat(file_path)
        data_cell = mat_data["data"]

        run_lengths = []
        run_channels = []

        for run_idx in range(data_cell.shape[1]):
            eeg = unwrap_public_run(data_cell[0, run_idx])
            run_lengths.append(int(eeg.shape[0]))
            run_channels.append(int(eeg.shape[1]))

            if run_idx == 0:
                plot_path = PLOTS_DIR / f"public_{file_path.stem}_run01.png"
                save_signal_plot(
                    eeg,
                    plot_path,
                    title=f"Public Dataset | {file_path.stem} | Run 1",
                )

        summary_rows.append(
            {
                "dataset": "public",
                "file_name": file_path.name,
                "subject": file_path.stem,
                "runs": int(data_cell.shape[1]),
                "mean_samples_per_run": float(np.mean(run_lengths)),
                "min_samples_per_run": int(np.min(run_lengths)),
                "max_samples_per_run": int(np.max(run_lengths)),
                "channels": int(run_channels[0]) if run_channels else None,
                "sampling_rate_hz": 250.0,
                "duration_per_run_seconds_mean": float(np.mean(run_lengths) / 250.0),
            }
        )

    return pd.DataFrame(summary_rows)


def summarize_lab_dataset() -> pd.DataFrame:
    summary_rows = []

    for file_path in sorted(LAB_DATASET_DIR.glob("*.mat")):
        with h5py.File(file_path, "r") as mat_file:
            if "y" not in mat_file:
                continue
            matrix = np.array(mat_file["y"])

        time_values = matrix[:, 0]
        eeg = matrix[:, 1:]
        sampling_rate = infer_sampling_rate(time_values)

        plot_path = PLOTS_DIR / f"lab_{file_path.stem}.png"
        save_signal_plot(
            eeg,
            plot_path,
            title=f"Lab Dataset | {file_path.stem}",
        )

        summary_rows.append(
            {
                "dataset": "lab",
                "file_name": file_path.name,
                "condition": extract_condition(file_path.stem),
                "samples": int(eeg.shape[0]),
                "channels": int(eeg.shape[1]),
                "sampling_rate_hz": sampling_rate,
                "duration_seconds": float(time_values[-1] - time_values[0]) if len(time_values) > 1 else 0.0,
            }
        )

    return pd.DataFrame(summary_rows)


def extract_condition(stem: str) -> str:
    if "(" in stem and ")" in stem:
        return stem.split("(")[-1].split(")")[0]
    return "unknown"


def save_signal_plot(eeg: np.ndarray, output_path: Path, title: str, max_channels: int = 4) -> None:
    channels_to_plot = min(max_channels, eeg.shape[1])
    sample_limit = min(2000, eeg.shape[0])
    time_axis = np.arange(sample_limit)

    fig, axes = plt.subplots(channels_to_plot, 1, figsize=(12, 2.5 * channels_to_plot), sharex=True)
    if channels_to_plot == 1:
        axes = [axes]

    for channel_idx in range(channels_to_plot):
        axes[channel_idx].plot(time_axis, eeg[:sample_limit, channel_idx], linewidth=0.8)
        axes[channel_idx].set_ylabel(f"Ch {channel_idx + 1}")
        axes[channel_idx].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Samples")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def build_combined_summary(public_df: pd.DataFrame, lab_df: pd.DataFrame) -> pd.DataFrame:
    public_view = public_df[
        ["dataset", "file_name", "subject", "runs", "channels", "sampling_rate_hz", "duration_per_run_seconds_mean"]
    ].rename(
        columns={
            "subject": "record_id",
            "runs": "segments",
            "duration_per_run_seconds_mean": "duration_seconds",
        }
    )

    lab_view = lab_df[
        ["dataset", "file_name", "condition", "samples", "channels", "sampling_rate_hz", "duration_seconds"]
    ].rename(
        columns={
            "condition": "record_id",
            "samples": "segments",
        }
    )

    return pd.concat([public_view, lab_view], ignore_index=True, sort=False)


def main() -> None:
    ensure_directories()

    public_df = summarize_public_dataset()
    lab_df = summarize_lab_dataset()
    combined_df = build_combined_summary(public_df, lab_df)

    public_df.to_csv(PHASE1_RESULTS_DIR / "public_dataset_summary.csv", index=False)
    lab_df.to_csv(PHASE1_RESULTS_DIR / "lab_dataset_summary.csv", index=False)
    combined_df.to_csv(PHASE1_RESULTS_DIR / "combined_dataset_summary.csv", index=False)

    print("Phase 1 complete.")
    print(f"Public records: {len(public_df)}")
    print(f"Lab records: {len(lab_df)}")
    print(f"Results saved under: {PHASE1_RESULTS_DIR}")


if __name__ == "__main__":
    main()

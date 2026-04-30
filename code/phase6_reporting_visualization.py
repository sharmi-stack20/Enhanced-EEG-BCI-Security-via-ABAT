from __future__ import annotations

import argparse

import matplotlib
import pandas as pd
import seaborn as sns

from integrated_common import PHASE4_ROOT, PHASE5_ROOT, PHASE6_ROOT, ensure_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PLOTS_DIR = PHASE6_ROOT / "plots"


def save_barplot(df: pd.DataFrame, x: str, y: str, title: str, output_path, hue: str | None = None, ylim=None) -> None:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x=x, y=y, hue=hue)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_heatmap(df: pd.DataFrame, title: str, output_path) -> None:
    plt.figure(figsize=(7, 6))
    sns.heatmap(df, annot=True, fmt=".0f", cmap="Blues")
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_dashboard(
    dataset_overview: pd.DataFrame,
    phase_progression: pd.DataFrame,
    comparison_df: pd.DataFrame,
    label_distribution: pd.DataFrame,
    output_path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Integrated BCI EEG ABAT Dashboard", fontsize=18)

    sns.barplot(data=dataset_overview, x="dataset", y="records", ax=axes[0, 0], palette="Blues_d")
    axes[0, 0].set_title("Dataset Record Counts")
    axes[0, 0].grid(True, alpha=0.25, axis="y")

    sns.barplot(data=comparison_df.melt(id_vars=["model"], value_vars=["accuracy", "balanced_accuracy", "macro_f1"], var_name="metric", value_name="value"),
                x="metric", y="value", hue="model", ax=axes[0, 1], palette="Set2")
    axes[0, 1].set_title("Baseline vs ABAT Metrics")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, alpha=0.25, axis="y")

    sns.lineplot(data=phase_progression, x="phase", y="count", hue="dataset", marker="o", linewidth=2.4, ax=axes[1, 0])
    axes[1, 0].set_title("Phase Progression Counts")
    axes[1, 0].grid(True, alpha=0.25)

    sns.barplot(data=label_distribution, x="label", y="count", hue="dataset", ax=axes[1, 1], palette="viridis")
    axes[1, 1].set_title("Class Distribution by Dataset")
    axes[1, 1].tick_params(axis="x", rotation=35)
    axes[1, 1].grid(True, alpha=0.25, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6: compare baseline vs ABAT for the integrated project.")
    parser.add_argument("--dataset", choices=["public", "lab"], default="lab")
    args = parser.parse_args()

    ensure_dir(PHASE6_ROOT)
    ensure_dir(PLOTS_DIR)
    baseline_df = pd.read_csv(PHASE4_ROOT / f"{args.dataset}_baseline_metrics.csv")
    abat_df = pd.read_csv(PHASE5_ROOT / f"{args.dataset}_abat_metrics.csv")
    comparison_df = pd.concat([baseline_df, abat_df], ignore_index=True, sort=False)
    comparison_df.to_csv(PHASE6_ROOT / f"{args.dataset}_baseline_vs_abat.csv", index=False)

    plot_df = comparison_df.melt(
        id_vars=["dataset", "model"],
        value_vars=["accuracy", "balanced_accuracy", "macro_f1"],
        var_name="metric",
        value_name="value",
    )

    save_barplot(
        plot_df,
        x="metric",
        y="value",
        hue="model",
        title=f"Integrated ABAT Comparison on {args.dataset.title()} EEG",
        output_path=PHASE6_ROOT / f"{args.dataset}_baseline_vs_abat.png",
        ylim=(0, 1),
    )

    baseline_cm = pd.read_csv(PHASE4_ROOT / f"{args.dataset}_baseline_confusion_matrix.csv", index_col=0)
    abat_cm = pd.read_csv(PHASE5_ROOT / f"{args.dataset}_abat_confusion_matrix.csv", index_col=0)
    save_heatmap(
        baseline_cm,
        title=f"{args.dataset.title()} Baseline Confusion Matrix",
        output_path=PLOTS_DIR / f"{args.dataset}_baseline_confusion_matrix.png",
    )
    save_heatmap(
        abat_cm,
        title=f"{args.dataset.title()} ABAT Confusion Matrix",
        output_path=PLOTS_DIR / f"{args.dataset}_abat_confusion_matrix.png",
    )

    dataset_overview = pd.DataFrame(
        [
            {
                "dataset": "public",
                "records": len(pd.read_csv(PHASE6_ROOT.parents[0] / "phase1_data_preparation" / "public_dataset_summary.csv")),
                "sampling_rate_hz": pd.read_csv(PHASE6_ROOT.parents[0] / "phase1_data_preparation" / "public_dataset_summary.csv")["sampling_rate_hz"].mean(),
            },
            {
                "dataset": "lab",
                "records": len(pd.read_csv(PHASE6_ROOT.parents[0] / "phase1_data_preparation" / "lab_dataset_summary.csv")),
                "sampling_rate_hz": pd.read_csv(PHASE6_ROOT.parents[0] / "phase1_data_preparation" / "lab_dataset_summary.csv")["sampling_rate_hz"].mean(),
            },
        ]
    )
    dataset_overview.to_csv(PHASE6_ROOT / "dataset_overview.csv", index=False)
    save_barplot(
        dataset_overview,
        x="dataset",
        y="sampling_rate_hz",
        title="Sampling Rate Comparison",
        output_path=PLOTS_DIR / "dataset_sampling_rates.png",
    )

    public_phase3 = pd.read_csv(PHASE6_ROOT.parents[0] / "phase3_feature_extraction" / "public_feature_summary.csv")
    lab_phase3 = pd.read_csv(PHASE6_ROOT.parents[0] / "phase3_feature_extraction" / "lab_feature_summary.csv")
    feature_overview = pd.DataFrame(
        [
            {"dataset": "public", "mean_windows": public_phase3["windows"].mean(), "mean_alpha_power": public_phase3["mean_alpha_power"].mean()},
            {"dataset": "lab", "mean_windows": lab_phase3["windows"].mean(), "mean_alpha_power": lab_phase3["mean_alpha_power"].mean()},
        ]
    )
    feature_overview.to_csv(PHASE6_ROOT / "feature_overview.csv", index=False)
    save_barplot(
        feature_overview,
        x="dataset",
        y="mean_windows",
        title="Average Windows per Record",
        output_path=PLOTS_DIR / "mean_windows_per_dataset.png",
    )

    phase_progression = pd.DataFrame(
        [
            {"dataset": "public", "phase": "Phase 1", "count": len(pd.read_csv(PHASE6_ROOT.parents[0] / "phase1_data_preparation" / "public_dataset_summary.csv"))},
            {"dataset": "lab", "phase": "Phase 1", "count": len(pd.read_csv(PHASE6_ROOT.parents[0] / "phase1_data_preparation" / "lab_dataset_summary.csv"))},
            {"dataset": "public", "phase": "Phase 2", "count": len(pd.read_csv(PHASE6_ROOT.parents[0] / "phase2_preprocessing" / "public_preprocessing_summary.csv"))},
            {"dataset": "lab", "phase": "Phase 2", "count": len(pd.read_csv(PHASE6_ROOT.parents[0] / "phase2_preprocessing" / "lab_preprocessing_summary.csv"))},
            {"dataset": "public", "phase": "Phase 3", "count": len(public_phase3)},
            {"dataset": "lab", "phase": "Phase 3", "count": len(lab_phase3)},
            {"dataset": "lab", "phase": "Phase 4", "count": float(baseline_df["accuracy"].iloc[0])},
            {"dataset": "lab", "phase": "Phase 5", "count": float(abat_df["accuracy"].iloc[0])},
        ]
    )
    phase_progression.to_csv(PHASE6_ROOT / "phase_progression_overview.csv", index=False)
    plt.figure(figsize=(11, 6))
    sns.lineplot(data=phase_progression[phase_progression["phase"].isin(["Phase 1", "Phase 2", "Phase 3"])], x="phase", y="count", hue="dataset", marker="o", linewidth=2.4)
    plt.title("Phase 1 to Phase 3 Progression by Dataset")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "phase_progression_counts.png", dpi=180)
    plt.close()

    public_labels = pd.Series([name[-1] for name in pd.read_csv(PHASE6_ROOT.parents[0] / "phase1_data_preparation" / "public_dataset_summary.csv")["subject"]], name="label")
    lab_labels = pd.read_csv(PHASE6_ROOT.parents[0] / "phase1_data_preparation" / "lab_dataset_summary.csv")["condition"]
    label_distribution = pd.concat(
        [
            public_labels.value_counts().rename_axis("label").reset_index(name="count").assign(dataset="public"),
            lab_labels.value_counts().rename_axis("label").reset_index(name="count").assign(dataset="lab"),
        ],
        ignore_index=True,
    )
    label_distribution.to_csv(PHASE6_ROOT / "label_distribution.csv", index=False)
    save_barplot(
        label_distribution,
        x="label",
        y="count",
        hue="dataset",
        title="Class Distribution by Dataset",
        output_path=PLOTS_DIR / "label_distribution.png",
    )

    save_dashboard(
        dataset_overview=dataset_overview,
        phase_progression=phase_progression,
        comparison_df=comparison_df,
        label_distribution=label_distribution,
        output_path=PLOTS_DIR / "integrated_project_dashboard.png",
    )

    report_lines = [
        f"# Integrated ABAT Report for {args.dataset.title()} Dataset",
        "",
        "## Baseline vs ABAT",
        "",
    ]
    for row in comparison_df.itertuples(index=False):
        epsilon_text = f", epsilon={row.epsilon}" if hasattr(row, "epsilon") and pd.notna(getattr(row, "epsilon", None)) else ""
        report_lines.append(
            f"- `{row.model}`: accuracy `{row.accuracy:.4f}`, balanced accuracy `{row.balanced_accuracy:.4f}`, macro F1 `{row.macro_f1:.4f}`{epsilon_text}"
        )
    report_lines.extend(
        [
            "",
            "## Reporting Outputs",
            "",
            "- Advanced comparison plots generated for metrics, confusion matrices, class distribution, phase progression, and dashboard view.",
            "- Dataset context from earlier phases has been folded into the final reporting layer for presentation use.",
        ]
    )

    (PHASE6_ROOT / f"{args.dataset}_abat_report.md").write_text("\n".join(report_lines) + "\n")

    print(f"Phase 6 complete for {args.dataset}.")
    print(comparison_df.to_string(index=False))
    print(f"\nSaved to: {PHASE6_ROOT}")


if __name__ == "__main__":
    main()

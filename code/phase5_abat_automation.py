from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from integrated_common import PHASE5_ROOT, build_window_dataset, ensure_dir, evaluate_predictions, stratified_group_split


def build_adversarial_samples(X: np.ndarray, epsilon: float) -> np.ndarray:
    # Lightweight adversarial-style augmentation for noisy EEG windows:
    # perturb each compact vector toward the sign of its standardized direction.
    direction = np.sign(X)
    return np.clip(X + epsilon * direction, -6.0, 6.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: ABAT-style adversarial augmentation and automation.")
    parser.add_argument("--dataset", choices=["public", "lab"], default="lab")
    parser.add_argument("--epsilon", type=float, default=0.15)
    args = parser.parse_args()

    ensure_dir(PHASE5_ROOT)
    X, y, groups, label_names, summary_df = build_window_dataset(args.dataset)
    train_mask, test_mask = stratified_group_split(groups, y)
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    X_adv = build_adversarial_samples(X_train, epsilon=args.epsilon)
    X_joint = np.vstack([X_train, X_adv])
    y_joint = np.concatenate([y_train, y_train])

    model = RandomForestClassifier(
        n_estimators=260,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_joint, y_joint)
    y_pred = model.predict(X_test)

    metrics, report_df, cm = evaluate_predictions(y_test, y_pred, label_names)
    metrics.update(
        {
            "dataset": args.dataset,
            "model": "abat_augmented_random_forest",
            "epsilon": args.epsilon,
            "samples": int(len(X)),
        }
    )

    summary_df.to_csv(PHASE5_ROOT / f"{args.dataset}_window_dataset_summary.csv", index=False)
    pd.DataFrame([metrics]).to_csv(PHASE5_ROOT / f"{args.dataset}_abat_metrics.csv", index=False)
    report_df.to_csv(PHASE5_ROOT / f"{args.dataset}_abat_report.csv")
    pd.DataFrame(cm, index=label_names, columns=label_names).to_csv(PHASE5_ROOT / f"{args.dataset}_abat_confusion_matrix.csv")

    print(f"Phase 5 complete for {args.dataset}.")
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"\nSaved to: {PHASE5_ROOT}")


if __name__ == "__main__":
    main()

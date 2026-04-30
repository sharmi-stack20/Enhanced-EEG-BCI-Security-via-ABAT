from __future__ import annotations

import argparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from integrated_common import PHASE4_ROOT, build_window_dataset, ensure_dir, evaluate_predictions, stratified_group_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: baseline model training for the integrated ABAT project.")
    parser.add_argument("--dataset", choices=["public", "lab"], default="lab")
    args = parser.parse_args()

    ensure_dir(PHASE4_ROOT)
    X, y, groups, label_names, summary_df = build_window_dataset(args.dataset)
    train_mask, test_mask = stratified_group_split(groups, y)
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics, report_df, cm = evaluate_predictions(y_test, y_pred, label_names)
    metrics.update({"dataset": args.dataset, "model": "baseline_random_forest", "samples": int(len(X))})

    summary_df.to_csv(PHASE4_ROOT / f"{args.dataset}_window_dataset_summary.csv", index=False)
    pd.DataFrame([metrics]).to_csv(PHASE4_ROOT / f"{args.dataset}_baseline_metrics.csv", index=False)
    report_df.to_csv(PHASE4_ROOT / f"{args.dataset}_baseline_report.csv")
    pd.DataFrame(cm, index=label_names, columns=label_names).to_csv(PHASE4_ROOT / f"{args.dataset}_baseline_confusion_matrix.csv")

    print(f"Phase 4 complete for {args.dataset}.")
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"\nSaved to: {PHASE4_ROOT}")


if __name__ == "__main__":
    main()

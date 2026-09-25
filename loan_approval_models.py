"""Compare the standard loan approval models with a fixed threshold."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from loan_approval_optimized import RANDOM_STATE, load_data, make_comparison_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare standard loan approval models."
    )
    parser.add_argument(
        "csv_file",
        type=Path,
        nargs="?",
        default=Path(__file__).with_name("loan_approval_dataset.csv"),
    )
    parser.add_argument(
        "--report-prefix",
        type=Path,
        default=Path(__file__).with_name("loan_model_comparison"),
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display the confusion-matrix chart after saving it.",
    )
    args = parser.parse_args()

    features, target = load_data(args.csv_file)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    models = {
        "Random Forest Baseline": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Support Vector Machine": SVC(probability=True, random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE),
    }
    metric_rows = []
    matrices = {}
    for model_name, model in models.items():
        pipeline = make_comparison_pipeline(X_train, model)
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.50).astype(int)
        matrix = confusion_matrix(y_test, predictions, labels=[1, 0])
        matrices[model_name] = matrix
        metric_rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Balanced Accuracy": balanced_accuracy_score(y_test, predictions),
                "Precision": precision_score(y_test, predictions, zero_division=0),
                "Recall": recall_score(y_test, predictions, zero_division=0),
                "F1-Score": f1_score(y_test, predictions, zero_division=0),
                "ROC AUC": roc_auc_score(y_test, probabilities),
            }
        )

    metrics = pd.DataFrame(metric_rows).set_index("Model")
    print("Standard model comparison threshold: 0.50")
    print("\nEvaluation Metrics")
    print(metrics.round(4).to_string())
    metrics.to_csv(args.report_prefix.with_name(f"{args.report_prefix.name}_metrics.csv"))

    with args.report_prefix.with_name(
        f"{args.report_prefix.name}_confusion_matrices.txt"
    ).open("w", encoding="utf-8") as report:
        report.write("Class order: [Approved, Rejected]\n")
        for model_name, matrix in matrices.items():
            report.write(f"\n{model_name}\n{matrix}\n")

    print("\nConfusion Matrices")
    print("Class order: [Approved, Rejected]")
    for model_name, matrix in matrices.items():
        print(f"\n{model_name}\n{matrix}")

    columns = 4
    rows = (len(matrices) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(16, rows * 4))
    axes = axes.flatten()
    for axis, (model_name, matrix) in zip(axes, matrices.items()):
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Greens",
            cbar=False,
            xticklabels=["Approved", "Rejected"],
            yticklabels=["Approved", "Rejected"],
            ax=axis,
        )
        axis.set_title(model_name)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("Actual label")
    for axis in axes[len(matrices):]:
        axis.remove()
    figure.suptitle("Standard Loan Model Confusion Matrices")
    figure.tight_layout()
    figure.savefig(
        args.report_prefix.with_name(
            f"{args.report_prefix.name}_confusion_matrices.png"
        ),
        dpi=150,
    )
    if args.show_plots:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()

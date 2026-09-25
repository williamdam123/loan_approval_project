"""Train and evaluate a tuned loan-approval classifier.

The test set remains untouched while the decision threshold is selected from
the validation set. This avoids inflating the reported evaluation metrics.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from loan_approval_utils import feature_importance_table

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_data(file_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()
    data = data.drop(
        columns=["loan_id", "approval_probability", "loan_decision"],
        errors="ignore",
    )
    for column in data.select_dtypes(include=["object", "category"]).columns:
        data[column] = data[column].str.strip()
    data["loan_status"] = data["loan_status"].str.strip().map(
        {"Approved": 1, "Rejected": 0}
    )
    if data["loan_status"].isna().any():
        raise ValueError("loan_status must contain only Approved or Rejected values.")
    return data.drop(columns=["loan_status"]), data["loan_status"].astype(int)


def make_pipeline(features: pd.DataFrame) -> Pipeline:
    categorical_columns = features.select_dtypes(include=["object", "category"]).columns
    numeric_columns = features.select_dtypes(
        exclude=["object", "category"]
    ).columns
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=1000,
        max_features=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def make_comparison_pipeline(features: pd.DataFrame, model) -> Pipeline:
    categorical_columns = features.select_dtypes(include=["object", "category"]).columns
    numeric_columns = features.select_dtypes(
        exclude=["object", "category"]
    ).columns
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def choose_threshold(probabilities, target) -> float:
    candidates = [index / 100 for index in range(30, 71)]
    return max(
        candidates,
        key=lambda threshold: (
            f1_score(target, probabilities >= threshold),
            accuracy_score(target, probabilities >= threshold),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an optimized loan model.")
    parser.add_argument(
        "csv_file",
        type=Path,
        nargs="?",
        default=Path(__file__).with_name("loan_approval_dataset.csv"),
    )
    parser.add_argument(
        "--report-prefix",
        type=Path,
        default=Path(__file__).with_name("loan_model_report"),
        help="Prefix for saved metrics, confusion matrices, and importance files.",
    )
    parser.add_argument(
        "--cross-validation",
        action="store_true",
        help="Run the lightweight three-fold Random Forest stability check.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display the confusion-matrix chart after saving it.",
    )
    args = parser.parse_args()

    features, target = load_data(args.csv_file)
    X_fit, X_test, y_fit, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    threshold_pipeline = make_pipeline(X_fit)
    out_of_fold_probabilities = cross_val_predict(
        threshold_pipeline,
        X_fit,
        y_fit,
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        method="predict_proba",
    )[:, 1]
    threshold = choose_threshold(out_of_fold_probabilities, y_fit)

    threshold_pipeline.fit(X_fit, y_fit)
    final_pipeline = threshold_pipeline
    test_probabilities = final_pipeline.predict_proba(X_test)[:, 1]
    predictions = (test_probabilities >= threshold).astype(int)

    model_predictions = {"Optimized Random Forest": predictions}
    model_probabilities = {"Optimized Random Forest": test_probabilities}
    comparison_models = {
        "Logistic Regression Optimized": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, random_state=RANDOM_STATE
        ),
        "Support Vector Machine": SVC(
            probability=True, random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
    }
    for model_name, model in comparison_models.items():
        comparison_pipeline = make_comparison_pipeline(X_fit, model)
        comparison_pipeline.fit(X_fit, y_fit)
        model_predictions[model_name] = comparison_pipeline.predict(X_test)
        model_probabilities[model_name] = comparison_pipeline.predict_proba(X_test)[:, 1]

    metric_rows = []
    matrices = {}
    for model_name, model_predictions_for_test in model_predictions.items():
        matrix = confusion_matrix(y_test, model_predictions_for_test, labels=[1, 0])
        true_positive, false_negative = matrix[0]
        false_positive, true_negative = matrix[1]
        metric_rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, model_predictions_for_test),
                "Balanced Accuracy": balanced_accuracy_score(
                    y_test, model_predictions_for_test
                ),
                "Precision": precision_score(
                    y_test, model_predictions_for_test, zero_division=0
                ),
                "Recall": recall_score(
                    y_test, model_predictions_for_test, zero_division=0
                ),
                "Specificity": true_negative / max(
                    true_negative + false_positive, 1
                ),
                "NPV": true_negative / max(true_negative + false_negative, 1),
                "F1-Score": f1_score(
                    y_test, model_predictions_for_test, zero_division=0
                ),
                "MCC": matthews_corrcoef(y_test, model_predictions_for_test),
                "ROC AUC": roc_auc_score(
                    y_test, model_probabilities[model_name]
                ),
                "PR AUC": average_precision_score(
                    y_test, model_probabilities[model_name]
                ),
                "Log Loss": log_loss(y_test, model_probabilities[model_name]),
            }
        )
        matrices[model_name] = matrix

    metrics = pd.DataFrame(metric_rows).set_index("Model")
    print(f"Optimized Random Forest threshold: {threshold:.2f}")
    print("\nEvaluation Metrics")
    print(metrics.round(4).to_string())

    cross_validation = pd.DataFrame()
    if args.cross_validation:
        cv = StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE)
        fold_scores = []
        for train_indices, validation_indices in cv.split(features, target):
            cv_pipeline = make_pipeline(features.iloc[train_indices])
            cv_pipeline.named_steps["model"].set_params(n_estimators=50)
            cv_pipeline.fit(features.iloc[train_indices], target.iloc[train_indices])
            fold_probabilities = cv_pipeline.predict_proba(
                features.iloc[validation_indices]
            )[:, 1]
            fold_predictions = (fold_probabilities >= 0.50).astype(int)
            fold_scores.append(
                {
                    "accuracy": accuracy_score(
                        target.iloc[validation_indices], fold_predictions
                    ),
                    "balanced_accuracy": balanced_accuracy_score(
                        target.iloc[validation_indices], fold_predictions
                    ),
                    "roc_auc": roc_auc_score(
                        target.iloc[validation_indices], fold_probabilities
                    ),
                }
            )
        scores = pd.DataFrame(fold_scores)
        cross_validation = pd.DataFrame(
            [
                {
                    "Model": "Optimized Random Forest",
                    "CV Accuracy Mean": scores["accuracy"].mean(),
                    "CV Accuracy Std": scores["accuracy"].std(),
                    "CV Balanced Accuracy Mean": scores["balanced_accuracy"].mean(),
                    "CV ROC AUC Mean": scores["roc_auc"].mean(),
                }
            ]
        ).set_index("Model")
        print("\nThree-Fold Cross-Validation")
        print(cross_validation.round(4).to_string())
    else:
        print("\nCross-validation skipped. Re-run with --cross-validation to enable it.")

    report_prefix = args.report_prefix
    metrics.to_csv(report_prefix.with_name(f"{report_prefix.name}_metrics.csv"))
    if not cross_validation.empty:
        cross_validation.to_csv(
            report_prefix.with_name(f"{report_prefix.name}_cross_validation.csv")
        )
    importance = feature_importance_table(final_pipeline, X_fit)
    importance.to_csv(
        report_prefix.with_name(f"{report_prefix.name}_feature_importance.csv"),
        index=False,
    )
    with report_prefix.with_name(f"{report_prefix.name}_confusion_matrices.txt").open(
        "w", encoding="utf-8"
    ) as report:
        report.write("Class order: [Approved, Rejected]\n")
        for model_name, matrix in matrices.items():
            report.write(f"\n{model_name}\n{matrix}\n")
    print(f"\nSaved evaluation reports with prefix {report_prefix}")
    if not importance.empty:
        print("\nTop Feature Importance")
        print(importance.head(10).round(4).to_string(index=False))

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
            cmap="Blues",
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

    figure.suptitle("Loan Approval Confusion Matrices")
    figure.tight_layout()
    figure.savefig(
        report_prefix.with_name(f"{report_prefix.name}_confusion_matrices.png"),
        dpi=150,
    )
    if args.show_plots:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()

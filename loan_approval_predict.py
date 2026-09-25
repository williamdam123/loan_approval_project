"""Predict approval decisions for new loan applications.

Run without arguments for interactive entry, or provide a separate CSV
containing the feature columns from loan_approval_dataset.csv.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from loan_approval_optimized import (
    load_data,
    make_pipeline,
)
from loan_approval_utils import (
    append_prediction_history,
    backup_file,
    explain_prediction,
    risk_category,
    validate_features,
)

DECISION_THRESHOLD = 0.5


def train_model(data_path: Path):
    features, target = load_data(data_path)
    pipeline = make_pipeline(features)
    pipeline.fit(features, target)
    return pipeline, features


def read_interactive_rows(features: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = set(
        features.select_dtypes(exclude=["object", "category"]).columns
    )
    while True:
        try:
            row_count = int(input("How many applications would you like to enter? "))
            if row_count > 0:
                break
        except ValueError:
            pass
        print("Please enter a positive whole number.")

    rows = []
    categorical_columns = features.select_dtypes(
        include=["object", "category"]
    ).columns
    allowed_values = {
        column: sorted(features[column].dropna().astype(str).unique())
        for column in categorical_columns
    }
    print("Enter every field using the same format as the training dataset.")
    print("Numbers must be whole numbers without commas, for example: 1000")
    for row_number in range(1, row_count + 1):
        print(f"\nApplication {row_number} of {row_count}")
        row = {}
        for column in features.columns:
            if column in categorical_columns:
                choices = " / ".join(allowed_values[column])
                prompt = f"{column} ({choices}): "
            else:
                prompt = f"{column} (whole number, no commas): "
            while True:
                value = input(prompt).strip()
                if not value:
                    print(f"{column} is required; enter a value.")
                    continue
                if column in categorical_columns:
                    matching_value = next(
                        (
                            option
                            for option in allowed_values[column]
                            if option.casefold() == value.casefold()
                        ),
                        None,
                    )
                    if matching_value is None:
                        choices = ", ".join(allowed_values[column])
                        print(f"Use one of: {choices}")
                        continue
                    row[column] = matching_value
                else:
                    if "," in value:
                        print("Do not use commas; enter 1000 instead of 1,000.")
                        continue
                    if not re.fullmatch(r"-?\d+", value):
                        print("Enter a whole number, such as 1000 or -100000.")
                        continue
                    row[column] = int(value)
                break
        rows.append(row)
    return pd.DataFrame(rows, columns=features.columns)


def load_input_rows(
    input_path: Path | None, features: pd.DataFrame
) -> pd.DataFrame:
    if input_path is None:
        rows = read_interactive_rows(features)
        validate_features(rows, features)
        return rows

    rows = pd.read_csv(input_path)
    rows.columns = rows.columns.str.strip()
    missing_columns = [column for column in features.columns if column not in rows]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Input CSV is missing these columns: {missing}")
    rows = rows[features.columns].copy()
    for column in features.select_dtypes(include=["object", "category"]).columns:
        allowed_values = set(features[column].dropna().astype(str).unique())
        rows[column] = rows[column].astype(str).str.strip()
        invalid_values = sorted(set(rows[column].dropna()) - allowed_values)
        if invalid_values:
            choices = ", ".join(sorted(allowed_values))
            raise ValueError(
                f"{column} contains invalid values {invalid_values}. "
                f"Use only: {choices}"
            )
    validate_features(rows, features)
    return rows


def save_applications(
    data_path: Path,
    input_rows: pd.DataFrame,
    statuses: pd.Series,
) -> None:
    data = pd.read_csv(data_path)
    data.columns = data.columns.str.strip()
    backup_path = backup_file(data_path)
    next_id = pd.to_numeric(data["loan_id"], errors="coerce").max() + 1
    new_rows = input_rows.copy()
    new_rows.insert(0, "loan_id", range(int(next_id), int(next_id) + len(new_rows)))
    new_rows["loan_status"] = statuses.map({1: "Approved", 0: "Rejected"}).values
    data = pd.concat([data, new_rows[data.columns]], ignore_index=True)
    data.to_csv(data_path, index=False)
    print(f"Backup created at {backup_path}")


def remove_applications(data_path: Path, application_ids: list[int]) -> None:
    data = pd.read_csv(data_path)
    data.columns = data.columns.str.strip()
    if "loan_id" not in data.columns:
        raise ValueError("The dataset does not contain a loan_id column.")

    existing_ids = pd.to_numeric(data["loan_id"], errors="coerce")
    missing_ids = sorted(set(application_ids) - set(existing_ids.dropna().astype(int)))
    if missing_ids:
        missing = ", ".join(str(application_id) for application_id in missing_ids)
        raise ValueError(f"These loan IDs were not found: {missing}")

    ids_to_remove = set(application_ids)
    rows_to_remove = existing_ids.isin(ids_to_remove)
    print("Applications selected for removal:")
    print(data.loc[rows_to_remove].to_string(index=False))
    answer = input("Permanently remove these applications? (y/n): ").strip().casefold()
    if answer not in {"y", "yes"}:
        print("No applications were removed.")
        return

    backup_path = backup_file(data_path)
    data.loc[~rows_to_remove].to_csv(data_path, index=False)
    print(f"Backup created at {backup_path}")
    print(f"Removed {rows_to_remove.sum()} application(s) from {data_path}")


def ask_to_remove_by_row_number(data_path: Path) -> None:
    data = pd.read_csv(data_path)
    total_applications = len(data)
    if total_applications == 0:
        print("There are no applications available to remove.")
        return

    answer = input(
        f"Would you like to remove applications from the file? "
        f"There are {total_applications} applications. (y/n): "
    ).strip().casefold()
    if answer not in {"y", "yes"}:
        return

    while True:
        value = input(
            f"Enter application number(s) from 1 to {total_applications}, "
            "separated by spaces or commas: "
        ).strip()
        try:
            row_numbers = [
                int(number)
                for number in value.replace(",", " ").split()
            ]
        except ValueError:
            row_numbers = []
        if row_numbers and all(1 <= number <= total_applications for number in row_numbers):
            break
        print(
            f"Enter whole numbers between 1 and {total_applications}."
        )

    row_indexes = sorted(set(number - 1 for number in row_numbers))
    print("Applications selected for removal:")
    print(data.iloc[row_indexes].to_string(index=False))
    confirmation = input("Permanently remove these applications? (y/n): ").strip().casefold()
    if confirmation not in {"y", "yes"}:
        print("No applications were removed.")
        return

    backup_path = backup_file(data_path)
    data.drop(index=row_indexes).to_csv(data_path, index=False)
    print(f"Backup created at {backup_path}")
    print(f"Removed {len(row_indexes)} application(s) from {data_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict loan approval for any number of applications."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        help="Separate applicant CSV; loan_status is optional for evaluation.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path for saving predictions.",
    )
    parser.add_argument(
        "--history-file",
        type=Path,
        default=Path(__file__).with_name("prediction_history.csv"),
        help="File where prediction history is appended.",
    )
    parser.add_argument(
        "--save-to-training-data",
        action="store_true",
        help="Append entered applications to the training CSV after prediction.",
    )
    parser.add_argument(
        "--training-data",
        type=Path,
        default=Path(__file__).with_name("loan_approval_dataset.csv"),
        help="Training dataset used to fit the model.",
    )
    parser.add_argument(
        "--remove-ids",
        nargs="+",
        type=int,
        metavar="LOAN_ID",
        help="Remove the listed loan IDs from the training CSV and exit.",
    )
    args = parser.parse_args()

    if args.remove_ids:
        remove_applications(args.training_data, args.remove_ids)
        return

    pipeline, training_features = train_model(args.training_data)
    input_rows = load_input_rows(args.input_csv, training_features)
    probabilities = pipeline.predict_proba(input_rows)[:, 1]
    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)

    results = input_rows.copy()
    results["approval_probability"] = [
        f"{probability:.2f}" for probability in probabilities
    ]
    results["loan_decision"] = [
        "Approved" if prediction else "Rejected" for prediction in predictions
    ]
    results["risk_category"] = [risk_category(probability) for probability in probabilities]
    results["explanation"] = [
        explain_prediction(row, probability)
        for (_, row), probability in zip(input_rows.iterrows(), probabilities)
    ]

    print(
        "Decision rule for every application: "
        f"probability >= {DECISION_THRESHOLD:.2f} means Approved; "
        f"probability < {DECISION_THRESHOLD:.2f} means Rejected."
    )
    print("\nLoan Predictions")
    print(results.to_string(index=False))
    append_prediction_history(results, args.history_file)
    print(f"Prediction history updated at {args.history_file}")

    save_to_training_data = args.save_to_training_data
    if not save_to_training_data:
        answer = input(
            f"Add these {len(input_rows)} applications to {args.training_data}? (y/n): "
        ).strip().casefold()
        save_to_training_data = answer in {"y", "yes"}

    if save_to_training_data:
        save_applications(
            args.training_data,
            input_rows,
            pd.Series(predictions),
        )
        print(f"Added applications to {args.training_data}")
    ask_to_remove_by_row_number(args.training_data)
    if args.output_csv is not None:
        results.to_csv(args.output_csv, index=False)
        print(f"\nSaved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()

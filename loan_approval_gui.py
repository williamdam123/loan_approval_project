"""Simple desktop form for one loan application."""

from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import pandas as pd

from loan_approval_predict import train_model
from loan_approval_utils import (
    append_prediction_history,
    explain_prediction,
    risk_category,
    validate_features,
)


class LoanApprovalApp:
    def __init__(self, root: tk.Tk, training_data: Path) -> None:
        self.root = root
        self.root.title("Loan Approval Predictor")
        self.pipeline, self.features = train_model(training_data)
        self.training_data = training_data
        self.fields: dict[str, tk.Widget] = {}
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind_all("<Control-c>", self.close)
        self.root.bind_all("<Control-q>", self.close)
        self.root.bind_all("<Escape>", self.close)
        self._build_form()

    def close(self, event=None) -> None:
        self.root.destroy()

    def _build_form(self) -> None:
        frame = ttk.Frame(self.root, padding=16)
        frame.grid(sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        for row_number, column in enumerate(self.features.columns):
            ttk.Label(frame, text=column).grid(
                row=row_number, column=0, sticky="w", padx=(0, 10), pady=3
            )
            if column in self.features.select_dtypes(include=["object", "category"]).columns:
                values = sorted(self.features[column].dropna().astype(str).unique())
                widget = ttk.Combobox(frame, values=values, state="readonly", width=28)
                if values:
                    widget.set(values[0])
            else:
                widget = ttk.Entry(frame, width=30)
            widget.grid(row=row_number, column=1, sticky="ew", pady=3)
            self.fields[column] = widget

        button_row = len(self.fields)
        ttk.Button(frame, text="Predict", command=self.predict).grid(
            row=button_row, column=0, columnspan=2, pady=(12, 6)
        )
        self.result = ttk.Label(frame, text="Enter an application and select Predict.", wraplength=500)
        self.result.grid(row=button_row + 1, column=0, columnspan=2, sticky="w")

    def predict(self) -> None:
        values = {}
        try:
            for column, widget in self.fields.items():
                value = widget.get().strip()
                if not value:
                    raise ValueError(f"{column} is required.")
                if column not in self.features.select_dtypes(include=["object", "category"]).columns:
                    value = int(value)
                values[column] = value
            row = pd.DataFrame([values], columns=self.features.columns)
            validate_features(row, self.features)
            probability = float(self.pipeline.predict_proba(row)[0, 1])
        except (TypeError, ValueError) as error:
            messagebox.showerror("Invalid application", str(error))
            return

        decision = "Approved" if probability >= 0.50 else "Rejected"
        category = risk_category(probability)
        explanation = explain_prediction(row.iloc[0], probability)
        result = pd.DataFrame(
            [
                {
                    **values,
                    "approval_probability": f"{probability:.2f}",
                    "loan_decision": decision,
                    "risk_category": category,
                    "explanation": explanation,
                }
            ]
        )
        append_prediction_history(result, self.training_data.with_name("prediction_history.csv"))
        self.result.configure(
            text=(
                f"Approval probability: {probability:.2f}\n"
                f"Decision: {decision}\n"
                f"Risk: {category}\n"
                f"Why: {explanation}"
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the loan approval desktop form.")
    parser.add_argument(
        "--training-data",
        type=Path,
        default=Path(__file__).with_name("loan_approval_dataset.csv"),
    )
    args = parser.parse_args()
    root = tk.Tk()
    LoanApprovalApp(root, args.training_data)
    root.mainloop()


if __name__ == "__main__":
    main()

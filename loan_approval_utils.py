"""Shared validation, explanation, history, and reporting helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from shutil import copy2

import pandas as pd


def validate_features(features: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Raise ValueError when applicant values are invalid or unsafe."""
    numeric_columns = reference.select_dtypes(exclude=["object", "category"]).columns
    nonnegative_columns = {
        "no_of_dependents",
        "income_annum",
        "loan_amount",
        "loan_term",
    }
    for column in numeric_columns:
        values = pd.to_numeric(features[column], errors="coerce")
        if values.isna().any() or not values.map(pd.api.types.is_number).all():
            raise ValueError(f"{column} must contain numeric values.")
        if column in nonnegative_columns and (values < 0).any():
            raise ValueError(f"{column} cannot contain negative values.")

    if "cibil_score" in features:
        cibil = pd.to_numeric(features["cibil_score"], errors="coerce")
        if ((cibil < 300) | (cibil > 900)).any():
            raise ValueError("cibil_score must be between 300 and 900.")
    if "loan_term" in features and (features["loan_term"] <= 0).any():
        raise ValueError("loan_term must be greater than zero.")

    for column in reference.select_dtypes(include=["object", "category"]).columns:
        allowed = set(reference[column].dropna().astype(str).str.strip())
        values = set(features[column].astype(str).str.strip())
        invalid = sorted(values - allowed)
        if invalid:
            choices = ", ".join(sorted(allowed))
            raise ValueError(f"{column} contains invalid values {invalid}. Use: {choices}")


def risk_category(probability: float) -> str:
    if probability >= 0.75:
        return "Low risk"
    if probability >= 0.50:
        return "Medium risk"
    return "High risk"


def explain_prediction(row: pd.Series, probability: float) -> str:
    factors = []
    cibil = float(row.get("cibil_score", 0))
    income = float(row.get("income_annum", 0))
    loan_amount = float(row.get("loan_amount", 0))
    if cibil >= 750:
        factors.append("strong CIBIL score")
    elif cibil < 650:
        factors.append("lower CIBIL score")
    if income and loan_amount / income <= 3:
        factors.append("loan amount is moderate compared with income")
    elif income and loan_amount / income > 5:
        factors.append("loan amount is high compared with income")
    assets = sum(float(row.get(column, 0)) for column in (
        "residential_assets_value", "commercial_assets_value",
        "luxury_assets_value", "bank_asset_value",
    ))
    if income and assets >= income:
        factors.append("substantial reported assets")
    elif income and assets < income / 2:
        factors.append("limited reported assets")
    if not factors:
        factors.append("the combined pattern of the supplied application features")
    direction = "supports approval" if probability >= 0.50 else "increases rejection risk"
    return f"{', '.join(factors)}; this {direction}."


def backup_file(path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.stem}.backup_{timestamp}{path.suffix}")
    copy2(path, backup_path)
    return backup_path


def append_prediction_history(results: pd.DataFrame, history_path: Path) -> None:
    history = results.copy()
    history.insert(0, "prediction_timestamp", datetime.now().isoformat(timespec="seconds"))
    if history_path.exists():
        history = pd.concat([pd.read_csv(history_path), history], ignore_index=True)
    history.to_csv(history_path, index=False)


def feature_importance_table(pipeline, features: pd.DataFrame) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["Feature", "Importance"])
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importance = pd.DataFrame({"Feature": names, "Importance": model.feature_importances_})
    importance["Feature"] = importance["Feature"].str.replace(
        r"^(numeric|categorical)__", "", regex=True
    )
    return importance.sort_values("Importance", ascending=False).reset_index(drop=True)

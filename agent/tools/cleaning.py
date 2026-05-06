# agent/tools/cleaning.py
"""
Data‑cleaning tools for the autonomous data‑science agent.
The main tool `clean_data` applies a dictionary of actions.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Union, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from langchain.tools import tool

from agent.tools import io_tools

# -------------------------------------------------------------------
# Helper: apply cleaning actions & return new summary
# -------------------------------------------------------------------
def _apply_cleaning(
    df: pd.DataFrame,
    drop_columns: Optional[List[str]] = None,
    drop_rows_with_na: Optional[List[str]] = None,
    fill_na: Optional[Dict[str, Union[str, int, float]]] = None,
    encode: Optional[Dict[str, str]] = None,
    scale: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Apply cleaning step‑by‑step. Returns the cleaned DataFrame.
    All steps are optional.
    """

    # 1. Drop columns
    if drop_columns:
        existing = [c for c in drop_columns if c in df.columns]
        df = df.drop(columns=existing)

    # 2. Drop rows where specific columns have NaN
    if drop_rows_with_na:
        df = df.dropna(subset=drop_rows_with_na)

    # 3. Fill missing values
    if fill_na:
        for col, strategy in fill_na.items():
            if col not in df.columns:
                continue
            if strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "mode":
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")
            elif isinstance(strategy, (int, float, str)):
                df[col] = df[col].fillna(strategy)
            # else ignore invalid strategy

    # 4. Encode categorical columns
    if encode:
        for col, method in encode.items():
            if col not in df.columns:
                continue
            if method == "onehot":
                # One‑hot encoding, drop first to avoid multicollinearity
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            elif method == "label":
                # Ordinal encoder for ordered categories or high cardinality
                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                df[col] = encoder.fit_transform(df[[col]]).astype(int)

    # 5. Scale numeric columns
    if scale:
        scaler = StandardScaler()
        # Only scale columns that are numeric and exist
        valid_cols = [c for c in scale if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if valid_cols:
            df[valid_cols] = scaler.fit_transform(df[valid_cols])

    return df

# -------------------------------------------------------------------
# Main cleaning tool
# -------------------------------------------------------------------
@tool
def clean_data(actions: str) -> str:
    """
    Apply a cleaning plan to the currently loaded DataFrame.
    `actions` must be a JSON string with (all optional):
    {
        "drop_columns": ["col1", "col2"],
        "drop_rows_with_na": ["col_target"],
        "fill_na": {"age": "median", "income": "mean", "city": "mode", "score": 0},
        "encode": {"gender": "onehot", "education": "label"},
        "scale": ["age", "income"]
    }
    Returns a summary of the cleaned data (shape, dtypes, missing).
    """
    df = io_tools._DF
    if df is None:
        if io_tools._PARQUET_PATH:
            df = pd.read_parquet(io_tools._PARQUET_PATH)
            io_tools._DF = df
        else:
            return json.dumps({"error": "No data loaded. Use load_csv first."})

    try:
        plan = json.loads(actions)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON in actions parameter."})

    # Record original shape
    orig_shape = df.shape

    df_cleaned = _apply_cleaning(
        df,
        drop_columns=plan.get("drop_columns"),
        drop_rows_with_na=plan.get("drop_rows_with_na"),
        fill_na=plan.get("fill_na"),
        encode=plan.get("encode"),
        scale=plan.get("scale"),
    )

    # Update global state
    io_tools._DF = df_cleaned

    # Save snapshot
    if io_tools._PARQUET_PATH:
        io_tools._DF.to_parquet(io_tools._PARQUET_PATH, index=False)

    # Build a concise summary
    new_shape = df_cleaned.shape
    missing = df_cleaned.isnull().sum().to_dict()
    dtypes = df_cleaned.dtypes.astype(str).to_dict()

    summary = {
        "original_shape": orig_shape,
        "new_shape": new_shape,
        "columns": df_cleaned.columns.tolist(),
        "dtypes": dtypes,
        "missing": missing,
        "actions_applied": plan,
    }
    return json.dumps(summary, indent=2, default=str)

@tool
def get_cleaning_recommendations() -> str:
    """
    Automatically generate a cleaning plan based on the current data,
    using simple heuristics. Returns a JSON plan that can be passed to clean_data.
    """
    df = io_tools._DF
    if df is None:
        if io_tools._PARQUET_PATH:
            df = pd.read_parquet(io_tools._PARQUET_PATH)
            io_tools._DF = df
        else:
            return json.dumps({"error": "No data loaded."})

    plan = {"drop_columns": [], "fill_na": {}, "encode": {}, "scale": []}
    for col in df.columns:
        missing_pct = df[col].isnull().mean()
        if missing_pct > 0.5:
            plan["drop_columns"].append(col)
        elif missing_pct > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                plan["fill_na"][col] = "median"
            else:
                plan["fill_na"][col] = "mode"

        # Encoding suggestion for object/categorical columns
        if df[col].dtype == object or df[col].nunique() < 20:
            if df[col].nunique() <= 5:
                plan["encode"][col] = "onehot"
            else:
                plan["encode"][col] = "label"

        # Scale numeric columns (heuristic: if range > 10)
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].dtype != 'bool':
            if df[col].max() - df[col].min() > 10:
                plan["scale"].append(col)

    # Remove empty sections
    plan = {k: v for k, v in plan.items() if v}
    return json.dumps(plan, indent=2, default=str)
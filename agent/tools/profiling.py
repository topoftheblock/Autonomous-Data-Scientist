# agent/tools/profiling.py
"""
Column‑profiling tools for the autonomous data‑science agent.
Each tool returns a structured JSON summary (and optionally a base64 plot).
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.tools import tool

# ------------------------------------------------------------
# Access the global dataframe held by the I/O tools module.
# This import happens at call time, so changes to io_tools._DF
# are always visible.
# ------------------------------------------------------------
from agent.tools import io_tools

# Plots are saved to disk and referenced by path rather than embedded as
# base64 in the tool response — inlining image bytes blows up the LLM's
# context window (a handful of profiled columns is enough to exceed 128k
# tokens) for data the model never needs to see pixel-for-pixel.
OUTPUT_DIR = "data/output"

def _ensure_output_dir():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def _numeric_plot(col_data: pd.Series, col_name: str) -> str:
    """Save a KDE+histogram plot for a numeric column and return its file path."""
    _ensure_output_dir()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(col_data.dropna(), kde=True, ax=ax)
    ax.set_title(f"Distribution of {col_name}")
    file_path = os.path.join(OUTPUT_DIR, f"profile_hist_{col_name}.png")
    fig.savefig(file_path, dpi=80)
    plt.close(fig)
    return file_path

def _categorical_plot(col_data: pd.Series, col_name: str, top_n: int = 15) -> str:
    """Save a count plot for a categorical column and return its file path."""
    value_counts = col_data.value_counts()
    if len(value_counts) > top_n:
        # Keep top categories and group the rest
        top = value_counts.head(top_n)
        other = pd.Series({"Other": value_counts.iloc[top_n:].sum()})
        plot_data = pd.concat([top, other])
    else:
        plot_data = value_counts

    _ensure_output_dir()
    fig, ax = plt.subplots(figsize=(max(6, len(plot_data)*0.4), 4))
    sns.barplot(x=plot_data.index, y=plot_data.values, ax=ax)
    ax.set_title(f"Frequency of {col_name}")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    file_path = os.path.join(OUTPUT_DIR, f"profile_count_{col_name}.png")
    fig.savefig(file_path, dpi=80)
    plt.close(fig)
    return file_path

# ------------------------------------------------------------
# Tools
# ------------------------------------------------------------
@tool
def profile_column(col: str) -> str:
    """
    Generate a detailed profile of a single column:
    - dtype, missing count, unique count, skew (if numeric)
    - for numeric: quantiles, mean, std, histogram (saved as PNG, path returned)
    - for categorical/object: top values, frequencies, count‑plot (saved as PNG, path returned)
    Use this to understand each column before cleaning or testing.
    """
    df = io_tools._DF
    if df is None:
        # Try automatic reload from parquet if path exists
        if io_tools._PARQUET_PATH:
            df = pd.read_parquet(io_tools._PARQUET_PATH)
            io_tools._DF = df
        else:
            return json.dumps({"error": "No data loaded. Use load_csv first."})

    if col not in df.columns:
        return json.dumps({"error": f"Column '{col}' not found. Available columns: {df.columns.tolist()}"})

    col_data = df[col]
    profile = {
        "column": col,
        "dtype": str(col_data.dtype),
        "missing": int(col_data.isnull().sum()),
        "missing_pct": round(col_data.isnull().mean() * 100, 2),
        "unique": int(col_data.nunique()),
    }

    # Numeric
    if pd.api.types.is_numeric_dtype(col_data):
        profile["type"] = "numeric"
        profile["stats"] = {
            "mean": round(col_data.mean(), 4),
            "std": round(col_data.std(), 4),
            "min": round(col_data.min(), 4),
            "25%": round(col_data.quantile(0.25), 4),
            "50%": round(col_data.median(), 4),
            "75%": round(col_data.quantile(0.75), 4),
            "max": round(col_data.max(), 4),
            "skew": round(col_data.skew(), 4),
        }
        profile["histogram_path"] = _numeric_plot(col_data, col)

    # Categorical, object, low cardinality numeric
    else:
        profile["type"] = "categorical"
        value_counts = col_data.value_counts().to_dict()
        # Stringify keys in case they are non‑serializable
        profile["value_counts"] = {str(k): v for k, v in list(value_counts.items())[:20]}
        profile["count_plot_path"] = _categorical_plot(col_data, col)

    return json.dumps(profile, indent=2, default=str)

@tool
def profile_all_columns() -> str:
    """
    Run profile_column on every column in the dataset and return a combined summary.
    Great for a first overview after loading data.
    """
    df = io_tools._DF
    if df is None:
        if io_tools._PARQUET_PATH:
            df = pd.read_parquet(io_tools._PARQUET_PATH)
            io_tools._DF = df
        else:
            return json.dumps({"error": "No data loaded."})

    profiles = {}
    for col in df.columns:
        # Call profile_column logic directly (not via @tool wrapper) to avoid overhead
        result_str = profile_column.invoke({"col": col})
        profiles[col] = json.loads(result_str)
    return json.dumps(profiles, indent=2, default=str)

@tool
def missing_summary() -> str:
    """
    Return a table of missing values for all columns (count and percentage).
    Useful for cleaning decisions.
    """
    df = io_tools._DF
    if df is None:
        if io_tools._PARQUET_PATH:
            df = pd.read_parquet(io_tools._PARQUET_PATH)
            io_tools._DF = df
        else:
            return json.dumps({"error": "No data loaded."})

    missing = pd.DataFrame({
        "column": df.columns,
        "missing": df.isnull().sum().values,
        "missing_pct": (df.isnull().mean() * 100).values.round(2)
    })
    missing = missing.sort_values("missing_pct", ascending=False)
    return missing.to_json(orient="records", indent=2)
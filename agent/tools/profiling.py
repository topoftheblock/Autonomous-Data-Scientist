# agent/tools/profiling.py
"""
Column‑profiling tools for the autonomous data‑science agent.
Each tool returns a structured JSON summary (and optionally a base64 plot).
"""

import pandas as pd
import numpy as np
import json
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.tools import tool

# ------------------------------------------------------------
# Access the global dataframe held by the I/O tools module.
# This import happens at call time, so changes to io_tools._DF
# are always visible.
# ------------------------------------------------------------
from agent.tools import io_tools

# ------------------------------------------------------------
# Helper: generate a PNG histogram as a base64 string
# ------------------------------------------------------------
def _numeric_plot(col_data: pd.Series, col_name: str) -> str:
    """Return a base64‑encoded PNG of a KDE+histogram plot for a numeric column."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(col_data.dropna(), kde=True, ax=ax)
    ax.set_title(f"Distribution of {col_name}")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def _categorical_plot(col_data: pd.Series, col_name: str, top_n: int = 15) -> str:
    """Return a base64‑encoded PNG of a count plot for a categorical column."""
    value_counts = col_data.value_counts()
    if len(value_counts) > top_n:
        # Keep top categories and group the rest
        top = value_counts.head(top_n)
        other = pd.Series({"Other": value_counts.iloc[top_n:].sum()})
        plot_data = pd.concat([top, other])
    else:
        plot_data = value_counts

    fig, ax = plt.subplots(figsize=(max(6, len(plot_data)*0.4), 4))
    sns.barplot(x=plot_data.index, y=plot_data.values, ax=ax)
    ax.set_title(f"Frequency of {col_name}")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ------------------------------------------------------------
# Tools
# ------------------------------------------------------------
@tool
def profile_column(col: str) -> str:
    """
    Generate a detailed profile of a single column:
    - dtype, missing count, unique count, skew (if numeric)
    - for numeric: quantiles, mean, std, histogram (base64)
    - for categorical/object: top values, frequencies, count‑plot (base64)
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
        profile["histogram_png"] = _numeric_plot(col_data, col)

    # Categorical, object, low cardinality numeric
    else:
        profile["type"] = "categorical"
        value_counts = col_data.value_counts().to_dict()
        # Stringify keys in case they are non‑serializable
        profile["value_counts"] = {str(k): v for k, v in list(value_counts.items())[:20]}
        profile["count_plot_png"] = _categorical_plot(col_data, col)

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
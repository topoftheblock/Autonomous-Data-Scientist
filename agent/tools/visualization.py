# agent/tools/visualization.py
"""
Visualisation tools for the autonomous data‑science agent.
All plots are saved as PNG files in the output directory.
"""

import pandas as pd
import numpy as np
import json
import os
import base64
import io
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.tools import tool

from agent.tools import io_tools

# -------------------------------------------------------------------
# Configurable output folder (can be overwritten by the graph)
# -------------------------------------------------------------------
OUTPUT_DIR = "data/output"

def _ensure_output_dir():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def _get_df():
    df = io_tools._DF
    if df is None and io_tools._PARQUET_PATH:
        df = pd.read_parquet(io_tools._PARQUET_PATH)
        io_tools._DF = df
    return df

# -------------------------------------------------------------------
# Individual plot tools
# -------------------------------------------------------------------
@tool
def plot_histogram(column: str, bins: int = 20) -> str:
    """
    Create a histogram with KDE for a numeric column.
    Saves the plot to data/output/histogram_{column}.png
    Returns the file path.
    """
    df = _get_df()
    if df is None:
        return json.dumps({"error": "No data loaded."})
    if column not in df.columns:
        return json.dumps({"error": f"Column '{column}' not found."})
    if not pd.api.types.is_numeric_dtype(df[column]):
        return json.dumps({"error": f"Column '{column}' is not numeric."})

    _ensure_output_dir()
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column].dropna(), bins=bins, kde=True)
    plt.title(f"Distribution of {column}")
    file_name = f"histogram_{column}.png"
    file_path = os.path.join(OUTPUT_DIR, file_name)
    plt.savefig(file_path)
    plt.close()
    return json.dumps({"plot": "histogram", "column": column, "path": file_path})

@tool
def plot_boxplot(column: str, by: str = "") -> str:
    """
    Create a boxplot for a numeric column, optionally grouped by a categorical column.
    Saves to data/output/boxplot_{column}[_by_{by}].png
    """
    df = _get_df()
    if df is None:
        return json.dumps({"error": "No data loaded."})
    if column not in df.columns:
        return json.dumps({"error": f"Column '{column}' not found."})
    if not pd.api.types.is_numeric_dtype(df[column]):
        return json.dumps({"error": f"Column '{column}' is not numeric."})

    _ensure_output_dir()
    plt.figure(figsize=(8, 5))
    if by and by in df.columns:
        sns.boxplot(x=by, y=column, data=df)
        plt.xticks(rotation=45)
        file_name = f"boxplot_{column}_by_{by}.png"
    else:
        sns.boxplot(y=df[column].dropna())
        file_name = f"boxplot_{column}.png"

    plt.title(f"Boxplot of {column}" + (f" by {by}" if by else ""))
    file_path = os.path.join(OUTPUT_DIR, file_name)
    plt.savefig(file_path)
    plt.close()
    return json.dumps({"plot": "boxplot", "column": column, "by": by if by else None, "path": file_path})

@tool
def plot_scatter(x: str, y: str, hue: str = "") -> str:
    """
    Create a scatter plot between two numeric columns.
    Optionally color points by a categorical column (hue).
    Saves to data/output/scatter_{x}_{y}[_hue_{hue}].png
    """
    df = _get_df()
    if df is None:
        return json.dumps({"error": "No data loaded."})
    for col in [x, y]:
        if col not in df.columns:
            return json.dumps({"error": f"Column '{col}' not found."})
        if not pd.api.types.is_numeric_dtype(df[col]):
            return json.dumps({"error": f"Column '{col}' is not numeric."})

    _ensure_output_dir()
    plt.figure(figsize=(7, 5))
    if hue and hue in df.columns:
        sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=0.6)
        file_name = f"scatter_{x}_{y}_hue_{hue}.png"
    else:
        sns.scatterplot(data=df, x=x, y=y, alpha=0.6)
        file_name = f"scatter_{x}_{y}.png"

    plt.title(f"{y} vs {x}")
    file_path = os.path.join(OUTPUT_DIR, file_name)
    plt.savefig(file_path)
    plt.close()
    return json.dumps({"plot": "scatter", "x": x, "y": y, "hue": hue if hue else None, "path": file_path})

@tool
def plot_correlation_heatmap(columns: str = "", method: str = "pearson") -> str:
    """
    Generate a correlation heatmap for given numeric columns (or all numeric if empty).
    Saves to data/output/correlation_heatmap.png
    """
    df = _get_df()
    if df is None:
        return json.dumps({"error": "No data loaded."})

    if columns:
        col_list = [c.strip() for c in columns.split(",") if c.strip() in df.columns]
    else:
        col_list = df.select_dtypes(include=np.number).columns.tolist()

    if len(col_list) < 2:
        return json.dumps({"error": "Need at least 2 numeric columns."})

    _ensure_output_dir()
    corr = df[col_list].corr(method=method)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title(f"{method.capitalize()} Correlation Heatmap")
    file_path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    plt.savefig(file_path)
    plt.close()
    return json.dumps({"plot": "correlation_heatmap", "columns": col_list, "path": file_path})

@tool
def plot_barplot(column: str, top_n: int = 15) -> str:
    """
    Create a horizontal bar plot for the top categories of a categorical column.
    Saves to data/output/barplot_{column}.png
    """
    df = _get_df()
    if df is None:
        return json.dumps({"error": "No data loaded."})
    if column not in df.columns:
        return json.dumps({"error": f"Column '{column}' not found."})

    _ensure_output_dir()
    counts = df[column].value_counts().head(top_n)
    plt.figure(figsize=(8, max(4, len(counts) * 0.3)))
    sns.barplot(x=counts.values, y=counts.index, orient="h")
    plt.title(f"Top {top_n} categories of {column}")
    plt.tight_layout()
    file_path = os.path.join(OUTPUT_DIR, f"barplot_{column}.png")
    plt.savefig(file_path)
    plt.close()
    return json.dumps({"plot": "barplot", "column": column, "top_n": top_n, "path": file_path})

# -------------------------------------------------------------------
# Quick multi‑view tool (optional)
# -------------------------------------------------------------------
@tool
def create_analysis_plots(columns: str = "") -> str:
    """
    Automatically create a set of standard plots for given columns
    (or all suitable columns if empty):
      - Histograms for numeric columns
      - Barplots for categorical columns
      - Correlation heatmap if more than 2 numeric columns exist.
    Returns a JSON list of file paths.
    """
    df = _get_df()
    if df is None:
        return json.dumps({"error": "No data loaded."})

    if columns:
        col_list = [c.strip() for c in columns.split(",") if c.strip() in df.columns]
    else:
        col_list = df.columns.tolist()

    plots = []
    # Numeric columns
    num_cols = [c for c in col_list if pd.api.types.is_numeric_dtype(df[c])]
    for col in num_cols:
        try:
            result_str = plot_histogram.invoke({"column": col})
            result = json.loads(result_str)
            if "path" in result:
                plots.append(result)
        except Exception:
            pass

    # Categorical columns
    cat_cols = [c for c in col_list if not pd.api.types.is_numeric_dtype(df[c])]
    for col in cat_cols:
        try:
            result_str = plot_barplot.invoke({"column": col})
            result = json.loads(result_str)
            if "path" in result:
                plots.append(result)
        except Exception:
            pass

    # Correlation heatmap
    if len(num_cols) > 2:
        try:
            result_str = plot_correlation_heatmap.invoke({"columns": ",".join(num_cols)})
            result = json.loads(result_str)
            if "path" in result:
                plots.append(result)
        except Exception:
            pass

    return json.dumps({"plots": plots}, indent=2)
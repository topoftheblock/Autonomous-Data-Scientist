# agent/tools/io_tools.py
"""
Data I/O tools for the autonomous data‑science agent.
These tools load/save data and provide summaries without exposing raw rows to the LLM.
"""

import pandas as pd
import json
import io
from pathlib import Path
from langchain.tools import tool

# -------------------------------------------------------------------
# Global state – persists across all tool calls during the pipeline.
# The executor node can initialise these before the agent runs.
# -------------------------------------------------------------------
_DF = None                 # Current DataFrame (loaded / cleaned)
_PARQUET_PATH = None       # Path to the serialised DataFrame (Parquet)

# -------------------------------------------------------------------
# Helper
# -------------------------------------------------------------------
def _df_to_summary(df: pd.DataFrame) -> str:
    """Convert a DataFrame into a JSON summary string without raw rows."""
    buf = io.StringIO()
    df.info(buf=buf)
    info_str = buf.getvalue()
    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "head": df.head(3).to_dict(orient="records"),
        "missing": df.isnull().sum().to_dict(),
        "describe": df.describe(include="all").to_dict(),
    }
    # Add a small info() string for quick reference
    summary["info"] = info_str
    return json.dumps(summary, indent=2, default=str)

# -------------------------------------------------------------------
# Tools
# -------------------------------------------------------------------
@tool
def load_csv(file_path: str) -> str:
    """
    Load a CSV file, store it globally, save a Parquet snapshot,
    and return a structured summary (shape, dtypes, missing values, head).
    """
    global _DF, _PARQUET_PATH
    df = pd.read_csv(file_path)
    _DF = df
    # Save a Parquet copy for serialisable passing and fast reload
    parquet_path = Path(file_path).with_suffix(".parquet")
    df.to_parquet(parquet_path, index=False)
    _PARQUET_PATH = str(parquet_path)
    return _df_to_summary(df)

@tool
def load_excel(file_path: str, sheet_name: str = 0) -> str:
    """
    Load an Excel file (optional). Same behaviour as load_csv.
    """
    global _DF, _PARQUET_PATH
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    _DF = df
    parquet_path = Path(file_path).with_suffix(".parquet")
    df.to_parquet(parquet_path, index=False)
    _PARQUET_PATH = str(parquet_path)
    return _df_to_summary(df)

@tool
def get_data_summary() -> str:
    """
    Return a structured summary of the *currently loaded* DataFrame.
    Use this after cleaning to inspect the new state.
    """
    global _DF
    if _DF is None:
        # Try to reload from the last saved Parquet if available
        if _PARQUET_PATH:
            _DF = pd.read_parquet(_PARQUET_PATH)
        else:
            return json.dumps({"error": "No data loaded. Use load_csv first."})
    return _df_to_summary(_DF)

@tool
def save_dataframe(path: str) -> str:
    """
    Save the current DataFrame to a Parquet file.
    Returns the path and shape of the saved data.
    """
    global _DF, _PARQUET_PATH
    if _DF is None:
        return json.dumps({"error": "No DataFrame to save."})
    _DF.to_parquet(path, index=False)
    _PARQUET_PATH = path
    return json.dumps({
        "message": "Data saved.",
        "path": path,
        "shape": _DF.shape,
    })

@tool
def reload_dataframe() -> str:
    """
    Reload the DataFrame from the last saved Parquet file.
    Useful after an external process wrote a new version.
    """
    global _DF, _PARQUET_PATH
    if _PARQUET_PATH:
        _DF = pd.read_parquet(_PARQUET_PATH)
        return _df_to_summary(_DF)
    return json.dumps({"error": "No Parquet path set. Load data first."})
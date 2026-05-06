# agent/tools/statistics.py
"""
Statistical‑analysis tools for the autonomous data‑science agent.
All tools work on the currently loaded DataFrame (via io_tools._DF).
"""

import pandas as pd
import numpy as np
import json
from scipy import stats
from langchain.tools import tool

from agent.tools import io_tools

# -------------------------------------------------------------------
# Helper: map test name → scipy function
# -------------------------------------------------------------------
_VALID_TESTS = [
    "ttest_ind",           # independent t‑test (numeric column across two groups)
    "mannwhitneyu",       # non‑parametric alternative to t‑test
    "oneway_anova",       # one‑way ANOVA (numeric across ≥2 groups)
    "kruskal",            # Kruskal‑Wallis (non‑parametric ANOVA)
    "chi2",               # chi‑square test of independence (two categorical)
    "pearsonr",           # Pearson correlation (two numeric)
    "spearmanr",          # Spearman rank correlation (two numeric / ordinal)
]

def _get_data_frame() -> pd.DataFrame:
    """Return the current DataFrame, loading from Parquet if necessary."""
    df = io_tools._DF
    if df is None and io_tools._PARQUET_PATH:
        df = pd.read_parquet(io_tools._PARQUET_PATH)
        io_tools._DF = df
    return df

# -------------------------------------------------------------------
# Main statistical‑test tool
# -------------------------------------------------------------------
@tool
def run_statistical_test(
    test_type: str,
    column1: str,
    column2: str = "",
    groupby: str = "",
    alternative: str = "two-sided"
) -> str:
    """
    Run a statistical test on the current dataset and return results.
    - test_type: one of {", ".join(_VALID_TESTS)}
    - column1: first column name
    - column2: second column name (for two‑sample tests or correlation)
    - groupby: categorical column to split the data into groups (for t‑test, ANOVA, etc.)
    - alternative: 'two-sided', 'less', or 'greater' (where applicable)
    
    Returns a JSON with test statistic, p‑value, and interpretation.
    """
    df = _get_data_frame()
    if df is None:
        return json.dumps({"error": "No data loaded."})

    if test_type not in _VALID_TESTS:
        return json.dumps({"error": f"Unknown test_type '{test_type}'. Choose from: {_VALID_TESTS}"})

    if column1 not in df.columns:
        return json.dumps({"error": f"Column '{column1}' not found."})

    # ---- Two‑sample / group comparisons (numeric vs categorical) ----
    if groupby and groupby in df.columns:
        if column2:  # Two columns + groupby is weird – warn but allow
            pass
        groups = df.groupby(groupby)[column1].apply(list).to_dict()
        unique_groups = list(groups.keys())
        if len(unique_groups) < 2:
            return json.dumps({"error": f"groupby column '{groupby}' must have at least 2 unique values."})

        if test_type == "ttest_ind":
            if len(unique_groups) != 2:
                return json.dumps({"error": "Independent t‑test requires exactly 2 groups."})
            a, b = groups[unique_groups[0]], groups[unique_groups[1]]
            # Remove NaN safely
            a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
            t_stat, p_val = stats.ttest_ind(a, b, alternative=alternative)
            result = {"test": test_type, "groups": unique_groups,
                      "statistic": t_stat, "p_value": p_val}

        elif test_type == "mannwhitneyu":
            if len(unique_groups) != 2:
                return json.dumps({"error": "Mann‑Whitney U requires exactly 2 groups."})
            a, b = groups[unique_groups[0]], groups[unique_groups[1]]
            a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
            u_stat, p_val = stats.mannwhitneyu(a, b, alternative=alternative)
            result = {"test": test_type, "groups": unique_groups,
                      "statistic": u_stat, "p_value": p_val}

        elif test_type == "oneway_anova":
            if len(unique_groups) < 2:
                return json.dumps({"error": "ANOVA requires at least 2 groups."})
            group_data = [pd.Series(groups[g]).dropna() for g in unique_groups]
            f_stat, p_val = stats.f_oneway(*group_data)
            result = {"test": test_type, "groups": unique_groups,
                      "statistic": f_stat, "p_value": p_val}

        elif test_type == "kruskal":
            if len(unique_groups) < 2:
                return json.dumps({"error": "Kruskal‑Wallis requires at least 2 groups."})
            group_data = [pd.Series(groups[g]).dropna() for g in unique_groups]
            h_stat, p_val = stats.kruskal(*group_data)
            result = {"test": test_type, "groups": unique_groups,
                      "statistic": h_stat, "p_value": p_val}

        else:
            return json.dumps({"error": f"Test '{test_type}' does not support groupby analysis."})

    # ---- Chi‑square (two categorical columns) ----
    elif test_type == "chi2":
        if not column2 or column2 not in df.columns:
            return json.dumps({"error": "Chi‑square test requires a second categorical column."})
        contingency = pd.crosstab(df[column1], df[column2])
        chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
        result = {"test": test_type, "statistic": chi2, "p_value": p_val, "dof": dof}

    # ---- Correlation (two numeric columns) ----
    elif test_type in ("pearsonr", "spearmanr"):
        if not column2 or column2 not in df.columns:
            return json.dumps({"error": f"{test_type} requires a second numeric column."})
        col1 = pd.to_numeric(df[column1], errors='coerce').dropna()
        col2 = pd.to_numeric(df[column2], errors='coerce').dropna()
        # Align by index
        common_idx = col1.index.intersection(col2.index)
        col1, col2 = col1[common_idx], col2[common_idx]
        if len(col1) < 3:
            return json.dumps({"error": "Not enough common non‑null values for correlation."})
        if test_type == "pearsonr":
            r, p_val = stats.pearsonr(col1, col2)
        else:
            r, p_val = stats.spearmanr(col1, col2)
        result = {"test": test_type, "statistic": r, "p_value": p_val}

    else:
        return json.dumps({"error": "Invalid test configuration."})

    # Interpret p‑value
    result["significant"] = p_val < 0.05
    if p_val < 0.001:
        interpretation = "Strong evidence against the null hypothesis."
    elif p_val < 0.05:
        interpretation = "Moderate evidence – reject null at 5% significance."
    else:
        interpretation = "No significant evidence – fail to reject null."
    result["interpretation"] = interpretation
    return json.dumps(result, indent=2, default=str)

# -------------------------------------------------------------------
# Correlation matrix
# -------------------------------------------------------------------
@tool
def correlation_matrix(
    columns: str = "",
    method: str = "pearson"
) -> str:
    """
    Compute a correlation matrix for numeric columns.
    - columns: comma‑separated list of columns (empty = all numeric)
    - method: 'pearson' or 'spearman'
    Returns a JSON with the matrix and a summary of strong correlations (|r| > 0.5).
    """
    df = _get_data_frame()
    if df is None:
        return json.dumps({"error": "No data loaded."})

    if method not in ("pearson", "spearman"):
        return json.dumps({"error": "Method must be 'pearson' or 'spearman'."})

    if columns:
        col_list = [c.strip() for c in columns.split(",") if c.strip() in df.columns]
    else:
        col_list = df.select_dtypes(include=np.number).columns.tolist()

    if len(col_list) < 2:
        return json.dumps({"error": "Need at least 2 numeric columns for correlation matrix."})

    corr_df = df[col_list].corr(method=method)
    # Convert to nested dict for JSON
    matrix_dict = corr_df.to_dict(orient="index")

    # Extract strong correlations (excluding self‑correlations)
    strong = []
    for i in range(len(col_list)):
        for j in range(i+1, len(col_list)):
            r = corr_df.iloc[i, j]
            if abs(r) > 0.5:
                strong.append({"pair": [col_list[i], col_list[j]], "coefficient": round(r, 4)})

    result = {"method": method, "matrix": matrix_dict, "strong_correlations": strong}
    return json.dumps(result, indent=2, default=str)

# -------------------------------------------------------------------
# Simple linear regression (scipy)
# -------------------------------------------------------------------
@tool
def linear_regression(target: str, predictors: str) -> str:
    """
    Run a simple linear regression (scipy.stats.linregress) for one predictor,
    or return coefficients for multiple predictors using numpy's lstsq.
    - target: dependent variable column name
    - predictors: comma‑separated independent variable column names
    Returns coefficients, R² (for single predictor), and p‑value.
    """
    df = _get_data_frame()
    if df is None:
        return json.dumps({"error": "No data loaded."})

    if target not in df.columns:
        return json.dumps({"error": f"Target column '{target}' not found."})
    pred_list = [c.strip() for c in predictors.split(",") if c.strip() in df.columns]
    if not pred_list:
        return json.dumps({"error": "No valid predictor columns."})

    # Drop rows with NaN in target or predictors
    data = df[[target] + pred_list].dropna()
    y = data[target].values.astype(float)
    X = data[pred_list].values.astype(float)

    if len(y) < 2:
        return json.dumps({"error": "Not enough non‑null data for regression."})

    # Single predictor – use linregress for more stats
    if len(pred_list) == 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(), y)
        result = {
            "intercept": intercept,
            "slope": slope,
            "r_squared": r_value**2,
            "p_value": p_value,
            "std_err": std_err,
            "predictor": pred_list[0],
            "target": target,
        }
    else:
        # Least‑squares solution (no intercept by default – add column of ones)
        X_with_const = np.column_stack([np.ones(len(y)), X])
        coeffs, residuals, rank, singular = np.linalg.lstsq(X_with_const, y, rcond=None)
        intercept, *slopes = coeffs.tolist()
        # Compute R²
        y_pred = X_with_const @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        result = {
            "intercept": intercept,
            "coefficients": {pred_list[i]: slopes[i] for i in range(len(pred_list))},
            "r_squared": r_squared,
            "target": target,
        }
    return json.dumps(result, indent=2, default=str)
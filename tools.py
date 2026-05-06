from langchain.tools import tool
import pandas as pd
import numpy as np
import json
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns

# Global variable to store the current dataframe (in production, use a context manager)
_DF = None

@tool
def load_csv(file_path: str) -> str:
    """Load a CSV file and return summary info: shape, columns, dtypes, first 3 rows, missing counts, descriptive stats."""
    global _DF
    _DF = pd.read_csv(file_path)
    buffer = io.StringIO()
    _DF.info(buf=buffer)
    info_str = buffer.getvalue()
    
    summary = {
        "shape": _DF.shape,
        "columns": _DF.columns.tolist(),
        "dtypes": _DF.dtypes.astype(str).to_dict(),
        "head": _DF.head(3).to_dict(orient="records"),
        "missing": _DF.isnull().sum().to_dict(),
        "describe": _DF.describe(include='all').to_dict()
    }
    return json.dumps(summary, indent=2, default=str)

@tool
def profile_column(column: str) -> str:
    """Generate detailed profile of a column: unique count, frequencies for categorical, histogram for numeric (base64 png)."""
    if _DF is None:
        return "No data loaded. Use load_csv first."
    col_data = _DF[column]
    result = {"column": column, "dtype": str(col_data.dtype), "missing": int(col_data.isnull().sum())}
    
    if col_data.dtype == 'object' or col_data.nunique() < 20:
        result["type"] = "categorical"
        result["unique_count"] = col_data.nunique()
        result["value_counts"] = col_data.value_counts().head(10).to_dict()
    else:
        result["type"] = "numeric"
        result["stats"] = col_data.describe().to_dict()
        # Generate histogram
        plt.figure(figsize=(6,4))
        sns.histplot(col_data.dropna(), kde=True)
        plt.title(f"Distribution of {column}")
        buf = io.BytesIO()
        plt.savefig(buf, format='png'); plt.close()
        buf.seek(0)
        result["histogram_png"] = base64.b64encode(buf.read()).decode()
    return json.dumps(result, indent=2, default=str)

@tool
def run_statistical_test(test_type: str, column1: str, column2: str = None, groupby: str = None) -> str:
    """Run common tests: 'ttest', 'anova', 'chi2', 'correlation'. Provide columns and optional groupby."""
    # Implementation using scipy.stats
    ...
    return json.dumps({"test": test_type, "p_value": p, "significant": p<0.05, "interpretation": "..."})
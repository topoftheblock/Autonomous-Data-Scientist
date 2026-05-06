# tests/test_tools.py
"""
Pytest tests for all agent tools.

Run from the project root:
    pytest tests/test_tools.py -v
"""

import json
import os
import sys
import tempfile
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Ensure the agent package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.tools import io_tools
from agent.tools import profiling
from agent.tools import cleaning
from agent.tools import statistics
from agent.tools import visualization
from agent.tools import sandbox

# -------------------------------------------------------------------
# Sample dataset fixture
# -------------------------------------------------------------------
@pytest.fixture
def sample_df():
    """Create a small, messy DataFrame for testing."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "name": ["Alice", "Bob", None, "Diana", "Eve", "Frank"],
        "age": [25, 30, np.nan, 22, 35, 28],
        "income": [50000, 60000, 75000, np.nan, 80000, 55000],
        "city": ["NY", "LA", "NY", "SF", "LA", None],
        "score": [88.5, 92.0, 85.0, np.nan, 91.0, 87.5],
        "target": [1, 1, 0, 0, 1, 0],
    })
    return df

@pytest.fixture(autouse=True)
def setup_global_df(sample_df, tmp_path):
    """Before each test, initialise io_tools globals with the sample data and a temp parquet path."""
    io_tools._DF = sample_df.copy()
    io_tools._PARQUET_PATH = str(tmp_path / "temp.parquet")
    io_tools._DF.to_parquet(io_tools._PARQUET_PATH, index=False)

# -------------------------------------------------------------------
# I/O Tools
# -------------------------------------------------------------------
class TestIOTools:
    def test_load_csv(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({"a": [1,2,3]})
        df.to_csv(csv_path, index=False)
        result = json.loads(io_tools.load_csv(str(csv_path)))
        assert result["shape"] == [3, 1]
        assert result["columns"] == ["a"]

    def test_get_data_summary(self):
        result = json.loads(io_tools.get_data_summary())
        assert result["shape"] == [6, 7]
        assert "age" in result["columns"]

    def test_save_and_reload_dataframe(self, tmp_path):
        new_path = tmp_path / "saved.parquet"
        result = json.loads(io_tools.save_dataframe(str(new_path)))
        assert result["shape"] == [6, 7]

        # Reload
        io_tools._DF = None
        result = json.loads(io_tools.reload_dataframe())
        assert result["shape"] == [6, 7]

# -------------------------------------------------------------------
# Profiling Tools
# -------------------------------------------------------------------
class TestProfiling:
    def test_profile_numeric_column(self):
        result = json.loads(profiling.profile_column("age"))
        assert result["type"] == "numeric"
        assert result["missing"] == 1
        assert "histogram_png" in result

    def test_profile_categorical_column(self):
        result = json.loads(profiling.profile_column("city"))
        assert result["type"] == "categorical"
        assert "count_plot_png" in result

    def test_profile_missing_column(self):
        result = json.loads(profiling.profile_column("nonexistent"))
        assert "error" in result

    def test_profile_all_columns(self):
        result = json.loads(profiling.profile_all_columns())
        assert len(result) == 7  # 7 columns

    def test_missing_summary(self):
        result = json.loads(profiling.missing_summary())
        assert len(result) == 7
        # "name" has 1 missing
        name_row = [r for r in result if r["column"] == "name"][0]
        assert name_row["missing"] == 1

# -------------------------------------------------------------------
# Cleaning Tools
# -------------------------------------------------------------------
class TestCleaning:
    def test_clean_data_drop_columns(self):
        plan = json.dumps({"drop_columns": ["id", "target"]})
        result = json.loads(cleaning.clean_data(plan))
        assert "id" not in result["columns"]
        assert result["original_shape"] == [6, 7]
        assert result["new_shape"] == [6, 5]

    def test_clean_data_fill_na(self):
        plan = json.dumps({"fill_na": {"age": "median", "city": "mode"}})
        result = json.loads(cleaning.clean_data(plan))
        assert result["missing"]["age"] == 0
        assert result["missing"]["city"] == 0

    def test_clean_data_encode_onehot(self):
        plan = json.dumps({"encode": {"city": "onehot"}})
        result = json.loads(cleaning.clean_data(plan))
        cols = result["columns"]
        assert "city" not in cols
        assert "city_LA" in cols or "city_NY" in cols

    def test_clean_data_scale(self):
        plan = json.dumps({"scale": ["age", "income"]})
        result = json.loads(cleaning.clean_data(plan))
        # After StandardScaler, mean ~0
        df = io_tools._DF
        assert abs(df["age"].mean()) < 1e-6

    def test_get_cleaning_recommendations(self):
        result = json.loads(cleaning.get_cleaning_recommendations())
        assert "fill_na" in result
        assert "encode" in result

# -------------------------------------------------------------------
# Statistical Tools
# -------------------------------------------------------------------
class TestStatistics:
    def test_ttest_ind(self):
        result = json.loads(statistics.run_statistical_test(
            test_type="ttest_ind", column1="age", groupby="target"
        ))
        assert "p_value" in result
        assert "significant" in result

    def test_chi2(self):
        result = json.loads(statistics.run_statistical_test(
            test_type="chi2", column1="city", column2="target"
        ))
        assert result["test"] == "chi2"

    def test_pearson_correlation(self):
        result = json.loads(statistics.run_statistical_test(
            test_type="pearsonr", column1="age", column2="income"
        ))
        assert -1 <= result["statistic"] <= 1

    def test_correlation_matrix(self):
        result = json.loads(statistics.correlation_matrix(
            columns="age,income,score", method="pearson"
        ))
        assert "matrix" in result
        assert "strong_correlations" in result

    def test_linear_regression_single(self):
        result = json.loads(statistics.linear_regression(
            target="score", predictors="age"
        ))
        assert "r_squared" in result
        assert "p_value" in result

    def test_invalid_test(self):
        result = json.loads(statistics.run_statistical_test(
            test_type="fake_test", column1="age"
        ))
        assert "error" in result

# -------------------------------------------------------------------
# Visualisation Tools (mock plt.savefig)
# -------------------------------------------------------------------
class TestVisualization:
    @patch("matplotlib.pyplot.savefig")
    def test_plot_histogram(self, mock_savefig):
        result = json.loads(visualization.plot_histogram("age"))
        assert "path" in result
        mock_savefig.assert_called_once()

    @patch("matplotlib.pyplot.savefig")
    def test_plot_boxplot(self, mock_savefig):
        result = json.loads(visualization.plot_boxplot("income", by="city"))
        assert "path" in result
        mock_savefig.assert_called_once()

    @patch("matplotlib.pyplot.savefig")
    def test_plot_correlation_heatmap(self, mock_savefig):
        result = json.loads(visualization.plot_correlation_heatmap(
            columns="age,income,score"
        ))
        assert "path" in result
        mock_savefig.assert_called_once()

    @patch("matplotlib.pyplot.savefig")
    def test_plot_barplot(self, mock_savefig):
        result = json.loads(visualization.plot_barplot("city"))
        assert "path" in result
        mock_savefig.assert_called_once()

    @patch("matplotlib.pyplot.savefig")
    def test_create_analysis_plots(self, mock_savefig):
        result = json.loads(visualization.create_analysis_plots())
        plots = result["plots"]
        assert len(plots) > 0
        assert all("path" in p for p in plots)

# -------------------------------------------------------------------
# Sandbox (RestrictedPython mode)
# -------------------------------------------------------------------
class TestSandbox:
    def test_simple_execution(self):
        code = "print(2 + 2)"
        result = json.loads(sandbox.execute_python(code))
        assert result["output"] == "4"

    def test_dataframe_access(self):
        code = """
import agent.tools.io_tools as io
df = io._DF
print(len(df))
"""
        result = json.loads(sandbox.execute_python(code))
        assert result["output"] == "6"

    def test_timeout(self, monkeypatch):
        monkeypatch.setattr(sandbox, "SANDBOX_TIMEOUT", 1)
        code = "while True: pass"
        result = json.loads(sandbox.execute_python(code))
        assert "timed out" in result["error"]

    def test_forbidden_open(self):
        code = "open('/etc/passwd')"
        result = json.loads(sandbox.execute_python(code))
        assert "error" in result
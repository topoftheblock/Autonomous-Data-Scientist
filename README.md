# Autonomous Data Scientist Agent

A fully hands-off AI agent that acts like a data scientist: it inspects data, cleans it, runs statistical analyses, creates visualizations, and compiles a final Markdown report — all without human intervention.

Powered by LLMs (GPT-4o, Claude, or local models) and built with LangGraph, this agent executes a complete data-science pipeline using safe, function-calling tools.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Tools Reference](#tools-reference)
- [Testing](#testing)
- [Security and Sandboxing](#security-and-sandboxing)
- [Limitations and Roadmap](#limitations-and-roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Autonomous end-to-end analysis** — No human input after you provide the CSV file.
- **Data profiling** — Every column inspected for type, missing values, distributions, and outliers.
- **Smart cleaning** — Automatic imputation, encoding, scaling, and column dropping based on heuristic rules.
- **Statistical tests** — t-test, ANOVA, Mann-Whitney, chi-square, Spearman/Pearson correlation, linear regression.
- **Visualizations** — Histograms, boxplots, scatter plots, bar plots, and correlation heatmaps saved as PNGs.
- **Self-healing** — If a step fails, the planner re-thinks the remaining tasks without human help.
- **Report generation** — All findings, test results, and plot references compiled into a clean Markdown report.
- **Modular and extensible** — Add your own tools, prompt templates, or swap the LLM.

---

## Architecture

The agent follows a hierarchical planner-executor pattern inside a LangGraph state machine:

```
┌─────────────┐
│  Ingestion  │  loads CSV → summary + Parquet
└──────┬──────┘
       │
┌──────▼──────┐
│   Planner   │  LLM creates step-by-step plan (JSON list)
└──────┬──────┘
       │
┌──────▼──────┐
│  Executor   │◄──┐  executes one plan step using tools
└──────┬──────┘   │
       │ success? │
┌──────▼──────┐   │
│   Report    │   │  compiles all results → final Markdown
└─────────────┘   │
                  │
           ┌──────┴──────┐
           │   Replan    │  LLM adjusts remaining plan after failure
           └─────────────┘
```

- **State** (`AgentState`) carries the plan, partial results, and error flags.
- **Nodes** are Python functions that read and write the state.
- **Tools** (decorated with `@tool`) are the data-science actions the LLM can call.

---

## Project Structure

```
data_scientist_agent/
├── agent/
│   ├── __init__.py
│   ├── state.py              # TypedDict AgentState
│   ├── graph.py              # LangGraph assembly & conditional edges
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── ingestion.py      # load CSV and store summary
│   │   ├── planner.py        # generate plan from summary
│   │   ├── executor.py       # execute one step (tool-calling agent)
│   │   ├── replan.py         # error recovery
│   │   └── reporter.py       # final markdown report
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── io_tools.py       # load_csv, save_dataframe, reload
│   │   ├── profiling.py      # profile_column, missing_summary
│   │   ├── cleaning.py       # clean_data, get_cleaning_recommendations
│   │   ├── statistics.py     # run_statistical_test, correlation_matrix, linear_regression
│   │   ├── visualization.py  # plot_histogram, boxplot, scatter, heatmap, barplot
│   │   └── sandbox.py        # safe Python execution (RestrictedPython / Docker)
│   ├── prompts/
│   │   ├── planner_prompt.py
│   │   ├── executor_prompt.py
│   │   └── replan_prompt.py
│   └── utils/
│       └── logging_config.py
├── config/
│   └── config.yaml           # model name, temperature, max iterations, etc.
├── data/
│   ├── input/                # drop your CSVs here
│   └── output/               # reports and plots appear here
├── tests/
│   ├── test_tools.py
│   ├── test_nodes.py
│   └── test_graph.py
├── notebooks/
│   └── exploration.ipynb     # manual testing playground
├── main.py                   # entry point
├── requirements.txt
└── README.md
```

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/autonomous-datascientist.git
   cd autonomous-datascientist
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   # .venv\Scripts\activate    # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set your API key**

   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

   For other LLMs, adjust `agent/graph.py` and `config/config.yaml` accordingly.

---

## Usage

**Basic run**

```bash
python main.py data/input/titanic.csv
```

This will create `data/output/report.md` and any plots in `data/output/`.

**Custom output path and recursion limit**

```bash
python main.py data/input/sales.csv --output reports/sales_analysis.md --recursion-limit 50
```

**Example console output**

```
================================================================================
                     AUTONOMOUS DATA SCIENCE REPORT
================================================================================
# Autonomous Data Science Report

## Execution Log

### Step 1
Profiled column 'age': numeric, missing 19.8%, mean=29.7, histogram saved.

### Step 2
Cleaned data: filled 'age' with median, dropped 'Cabin', one-hot encoded 'Sex'.

... (further steps)

## Statistical Tests
- t-test (Survived vs Age): p=0.032 → significant
- Chi-square (Sex vs Survived): p<0.001 → significant

## Visualizations
![Age distribution](data/output/histogram_age.png)
...
================================================================================
```

The agent runs entirely on its own — you only wait for the final report.

---

## Configuration

All tunable parameters live in `config/config.yaml` or as environment variables.

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `gpt-4o` | LLM model for planner and executor |
| `temperature` | `0.0` | Determinism (0 = completely deterministic) |
| `max_executor_iterations` | `10` | Max tool calls per step |
| `recursion_limit` | `30` | Max LangGraph steps |
| `sandbox_mode` | `restrictedpython` | `restrictedpython` or `docker` |
| `sandbox_timeout` | `30` | Seconds allowed for arbitrary code |
| `allowed_modules` | `pandas,numpy,...` | Modules available inside the sandbox |

Environment variables override the config file. The most important one is `OPENAI_API_KEY`.

---

## How It Works

1. **Ingestion** — `main.py` loads the CSV, saves a Parquet snapshot, and creates a structured summary (shape, dtypes, missing counts).
2. **Planning** — The Planner LLM sees only the summary and writes a detailed JSON plan: "Profile column 'age'", "Impute missing 'age' with median", "Run t-test comparing 'fare' by 'pclass'", and so on.
3. **Step Execution (looping)** — The Executor receives one step at a time. It has access to all tools (`profile_column`, `clean_data`, `run_statistical_test`, etc.) and may call several to finish the task. After completion it signals `DONE` with a human-readable summary.
4. **Replanning** — If a step fails (e.g., column not found, inappropriate test), the Replan LLM modifies the remaining plan.
5. **Reporting** — When all steps are done, the Reporter compiles all step summaries into a Markdown file with embedded plot references.

No human is involved after pressing Enter.

---

## Tools Reference

### I/O

| Tool | Description |
|---|---|
| `load_csv(file_path)` | Load CSV, return summary, save Parquet |
| `get_data_summary()` | Quick overview of current DataFrame |
| `save_dataframe(path)` | Persist cleaned data |
| `reload_dataframe()` | Reload from last saved Parquet |

### Profiling

| Tool | Description |
|---|---|
| `profile_column(col)` | Detailed stats and base64 histogram / count plot |
| `profile_all_columns()` | Profiles every column at once |
| `missing_summary()` | Table of missing values per column |

### Cleaning

| Tool | Description |
|---|---|
| `clean_data(actions)` | Apply a JSON cleaning plan (drop, fill, encode, scale) |
| `get_cleaning_recommendations()` | Heuristic cleaning suggestions |

### Statistics

| Tool | Description |
|---|---|
| `run_statistical_test(...)` | t-test, ANOVA, chi-square, Mann-Whitney, correlation |
| `correlation_matrix(...)` | Matrix and strong correlation list |
| `linear_regression(target, predictors)` | scipy-based regression |

### Visualization

| Tool | Description |
|---|---|
| `plot_histogram(col)` | Save histogram PNG |
| `plot_boxplot(col, by)` | Boxplot, optionally grouped |
| `plot_scatter(x, y, hue)` | Scatter plot |
| `plot_correlation_heatmap(cols)` | Annotated heatmap |
| `plot_barplot(col, top_n)` | Top-N categories bar chart |
| `create_analysis_plots(cols)` | Auto-generate all standard plots |

### Sandbox

| Tool | Description |
|---|---|
| `execute_python(code)` | Run arbitrary (safe) Python in a sandbox |

---

## Testing

The test suite covers all tools and the full graph logic (using mocked LLMs). Run it with:

```bash
pytest tests/ -v
```

- `test_tools.py` — Validates every tool against a sample DataFrame.
- `test_nodes.py` — Checks planner, executor, replan, and reporter behaviour.
- `test_graph.py` — End-to-end integration with a real CSV and mocked LLMs.

Tests are isolated and do not require an API key.

---

## Security and Sandboxing

- **No raw data exposure** — The LLM sees only aggregated summaries, never individual rows.
- **Restricted Python** — The `execute_python` sandbox uses RestrictedPython to disable dangerous builtins (`open`, `exec`, `eval`, `__import__`, etc.).
- **Timeout** — Arbitrary code is killed after `sandbox_timeout` seconds.
- **Docker isolation (optional)** — Set `SANDBOX_MODE=docker` to run code in a container with no network access and limited resources.

---

## Limitations and Roadmap

**Current limitations**

- **Domain knowledge** — The agent applies generic data-science heuristics; it does not understand industry-specific nuances unless injected into the prompts.
- **Visualization interpretation** — The LLM sees only base64-encoded plots (in profiling) or file paths (final report); it does not visually interpret images.
- **Large datasets** — All data is kept in memory; for big data, add sampling or incremental processing.
- **Multi-file support** — Currently only a single CSV or Excel file is handled.

**Planned improvements**

- LLM-driven feature engineering
- Time-series analysis support
- Docker-based sandbox integration
- Web UI (Streamlit) for starting runs interactively

---

## Contributing

Contributions are welcome. Please open an issue to discuss what you would like to change, or submit a pull request directly.

Before submitting a PR, make sure tests pass:

```bash
pytest tests/ -v
```

---

## License

MIT License. See `LICENSE` for details.
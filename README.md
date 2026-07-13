# Autonomous Data Scientist Agent

A fully hands-off AI agent that acts like a data scientist: it inspects a CSV, cleans it, runs statistical tests, creates visualizations, and compiles a final Markdown report — all without human intervention.

Built with **LangGraph** + **LangChain** (OpenAI function-calling tools), the agent runs a planner/executor/replanner loop until the analysis is complete, then writes everything to a single report.

**[See a real, unedited run on the Titanic dataset →](examples/titanic_report.md)**

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
- [License](#license)

---

## Features

- **Autonomous end-to-end analysis** — no human input after you provide the CSV file.
- **Data profiling** — every column inspected for type, missing values, distributions, and outliers.
- **Smart cleaning** — imputation, one-hot encoding, scaling, and column dropping, driven by the agent's own judgment.
- **Statistical tests** — t-test, ANOVA, Mann-Whitney, chi-square, Spearman/Pearson correlation, linear regression.
- **Visualizations** — histograms, boxplots, scatter plots, bar plots, and correlation heatmaps saved as PNGs.
- **Self-healing** — if a step fails, a replanner rewrites the remaining plan without human help.
- **Sandboxed code execution** — the agent can also run arbitrary Python via a RestrictedPython sandbox for anything the built-in tools don't cover.
- **Report generation** — all findings, test results, and plot paths compiled into a Markdown report.

---

## Architecture

The agent follows a planner → executor → (replanner) → reporter pattern inside a LangGraph state machine:

```
┌─────────────┐
│  Ingestion  │  loads CSV → summary + Parquet snapshot
└──────┬──────┘
       │
┌──────▼──────┐
│   Planner   │  LLM creates a step-by-step plan (JSON list)
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
            │   Replan    │  LLM adjusts the remaining plan after a failure
            └─────────────┘
```

- **State** (`AgentState`, in `agent/state.py`) carries the plan, accumulated step results, and error flags. Fields like `step_results` use LangGraph's `operator.add` reducer, so every node returns a small **partial** update dict rather than mutating and returning the whole state.
- **Nodes** (`agent/nodes/`) are plain functions that read the state and return updates.
- **Tools** (`agent/tools/`, decorated with `@tool`) are the data-science actions the executor LLM can call.

---

## Project Structure

```
Autonomous-Data-Scientist/
├── agent/
│   ├── state.py                 # TypedDict AgentState
│   ├── graph.py                 # LangGraph assembly & conditional edges
│   ├── nodes/
│   │   ├── ingestion_node.py    # load CSV, store summary
│   │   ├── planner_node.py      # generate plan from summary
│   │   ├── execution_step.py    # execute one step (tool-calling agent)
│   │   ├── replan_node.py       # error recovery
│   │   └── reporter_node.py     # final markdown report
│   └── tools/
│       ├── io_tools.py          # load_csv, save_dataframe, reload, get_dataframe
│       ├── profiling.py         # profile_column, profile_all_columns, missing_summary
│       ├── cleaning.py          # clean_data, get_cleaning_recommendations
│       ├── statistics.py        # run_statistical_test, correlation_matrix, linear_regression
│       ├── visualization.py     # plot_histogram, boxplot, scatter, heatmap, barplot
│       └── sandbox.py           # sandboxed arbitrary Python execution (RestrictedPython)
├── config/
│   └── config.yaml              # documents the tunable parameters (see Configuration below)
├── data/
│   ├── input/                   # sample CSVs (e.g. titanic.csv)
│   └── output/                  # reports and plots land here (gitignored)
├── examples/
│   └── titanic_report.md        # a real, unedited run against data/input/titanic.csv
├── tests/
│   ├── test_tools.py            # every tool, against a small synthetic DataFrame
│   ├── test_models.py           # planner/executor/replan/reporter node logic (mocked LLM)
│   └── test_graph.py            # full graph integration, including the replan path (mocked LLM)
├── notebook/
│   └── exploration.ipynb        # manual scratchpad
├── main.py                      # CLI entry point
└── requirements.txt
```

---

## Installation

1. **Clone the repository and enter it**

   ```bash
   git clone <your-fork-url>
   cd Autonomous-Data-Scientist
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   # .venv\Scripts\activate    # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Requires Python 3.10+. On Linux/macOS the sandbox uses `fork()`-based
   multiprocessing to enforce timeouts; this doesn't work on Windows, so the
   sandbox tool (`execute_python`) is best-effort there.

4. **Set your API key**

   ```bash
   export OPENAI_API_KEY="sk-..."
   # or put it in a .env file — main.py loads it automatically via python-dotenv
   ```

---

## Usage

**Basic run**

```bash
python main.py data/input/titanic.csv
```

This creates `data/output/report.md` plus any plots the agent generated, and also prints the report to stdout.

**Custom output path and step budget**

```bash
python main.py data/input/sales.csv --output reports/sales_analysis.md --recursion-limit 40
```

`--recursion-limit` caps how many plan steps (including replans) the LangGraph
run may take before it stops and compiles whatever results it has so far —
see [`examples/titanic_report.md`](examples/titanic_report.md) for a complete
real run (profiling → cleaning → two statistical tests → two plots → report).

---

## Configuration

Most run-time knobs are plain Python constants or environment variables read directly by the tool modules — `config/config.yaml` documents the same values as a single reference point, but nothing in the code parses that file yet (see Roadmap).

| Parameter | Where it lives | Default |
|---|---|---|
| LLM model | `ChatOpenAI(model=...)` in `planner_node.py` / `execution_step.py` | `gpt-4o-mini` |
| `--recursion-limit` | CLI flag / `main.py` | `25` |
| Executor `max_iterations` | `execution_step.py` | `10` |
| `SANDBOX_MODE` | env var, `sandbox.py` | `restrictedpython` |
| `SANDBOX_TIMEOUT` | env var, `sandbox.py` | `30` seconds |
| `SANDBOX_ALLOWED_MODULES` | env var, `sandbox.py` | `pandas,numpy,matplotlib,seaborn,scipy,sklearn,json,csv,datetime,math,statistics` |

The most important environment variable is `OPENAI_API_KEY`.

---

## How It Works

1. **Ingestion** — `ingestion_node` loads the CSV, saves a Parquet snapshot, and builds a structured JSON summary (shape, dtypes, missing counts, a small head sample) — never the raw rows.
2. **Planning** — `planner_node` sees only that summary and writes a short JSON plan (aims for 8-12 steps): profile, clean, run at least two statistical tests, create at least two plots, generate the report.
3. **Step execution (looping)** — `execution_step` hands one step at a time to a tool-calling agent with access to all tools. It signals `DONE` plus a short summary when finished; that summary (not the raw tool output) becomes the step's entry in `step_results` and is what subsequent steps see as history.
4. **Replanning** — if a step raises an exception, `replan_node` asks the LLM to rewrite the remaining plan; if that itself fails to parse, it falls back to an "abort and report with available results" step so the pipeline always terminates with *something*.
5. **Reporting** — `reporter_node` compiles every step's summary, the cleaning/statistics/plot metadata, and writes the final Markdown file.

If the LangGraph step budget (`--recursion-limit`) is hit before reaching the report step, `main.py` catches the resulting `GraphRecursionError` and runs the reporter manually against whatever was accomplished, so a run never dies with nothing to show for it.

---

## Tools Reference

### I/O (`agent/tools/io_tools.py`)

| Tool | Description |
|---|---|
| `load_csv(file_path)` | Load a CSV, return a summary, save a Parquet snapshot |
| `get_data_summary()` | Quick overview of the currently loaded DataFrame |
| `save_dataframe(path)` | Persist the current DataFrame to Parquet |
| `reload_dataframe()` | Reload from the last saved Parquet path |
| `get_dataframe()` | Plain Python accessor (not an LLM tool) used by sandboxed code — `df._DF` would be blocked by RestrictedPython's underscore-attribute guard |

### Profiling (`agent/tools/profiling.py`)

| Tool | Description |
|---|---|
| `profile_column(col)` | Type, missing %, quantiles/skew (numeric) or top values (categorical); saves a plot PNG and returns its path |
| `profile_all_columns()` | Runs `profile_column` over every column |
| `missing_summary()` | Table of missing counts/percentages per column |

### Cleaning (`agent/tools/cleaning.py`)

| Tool | Description |
|---|---|
| `clean_data(actions)` | Apply a JSON cleaning plan: `drop_columns`, `fill_na`, `encode`, `scale` |
| `get_cleaning_recommendations()` | Heuristic suggestions based on missingness/cardinality |

### Statistics (`agent/tools/statistics.py`)

| Tool | Description |
|---|---|
| `run_statistical_test(...)` | t-test, ANOVA, Mann-Whitney, chi-square, Pearson/Spearman correlation |
| `correlation_matrix(...)` | Full matrix plus a list of strong correlations |
| `linear_regression(target, predictors)` | Simple/multiple linear regression via scipy/sklearn |

### Visualization (`agent/tools/visualization.py`)

| Tool | Description |
|---|---|
| `plot_histogram(col)` | Histogram + KDE, saved as PNG |
| `plot_boxplot(col, by)` | Boxplot, optionally grouped |
| `plot_scatter(x, y, hue)` | Scatter plot |
| `plot_correlation_heatmap(cols)` | Annotated correlation heatmap |
| `plot_barplot(col, top_n)` | Top-N category bar chart |
| `create_analysis_plots(cols)` | Generates the standard set of plots in one call |

### Sandbox (`agent/tools/sandbox.py`)

| Tool | Description |
|---|---|
| `execute_python(code)` | Runs arbitrary Python under RestrictedPython, with a guarded `import` allow-list and a hard timeout; use `agent.tools.io_tools.get_dataframe()` to reach the current DataFrame |

---

## Testing

The test suite covers every tool plus the full graph logic (LLM calls mocked, so no API key is needed):

```bash
pip install pytest
pytest tests/ -v
```

- `test_tools.py` — every tool against a small synthetic DataFrame, including the RestrictedPython sandbox (import allow-listing, `df['col']` indexing, timeouts, forbidden builtins).
- `test_models.py` — planner/executor/replan/reporter node behavior in isolation.
- `test_graph.py` — full graph integration on a real tiny CSV, including the failure → replan → recovery path.

All 40 tests pass as of this writing.

---

## Security and Sandboxing

- **No raw data exposure** — the LLM only ever sees aggregated summaries and small samples, never the full dataset.
- **Restricted Python** — `execute_python` compiles code with RestrictedPython, which strips dangerous builtins (`open`, `eval`, raw `__import__`, underscore-attribute access) and requires explicit guards for attribute/item access, iteration, and in-place operators (all wired up in `sandbox.py`). A custom guarded `__import__` only allows the modules already on the allow-list.
- **Timeout** — sandboxed code runs in a forked subprocess and is killed after `SANDBOX_TIMEOUT` seconds.
- **Docker isolation (not yet implemented)** — `SANDBOX_MODE=docker` is a documented but unimplemented stub in `sandbox.py`; the default `restrictedpython` mode is what's actually enforced today.

---

## Limitations and Roadmap

**Current limitations**

- **`config/config.yaml` isn't wired up yet** — it documents intended settings, but the code currently reads model names/timeouts from constants and environment variables instead.
- **Step budget vs. plan length** — very wide datasets (many columns) can eat the step budget on profiling alone if the planner doesn't batch; the planner prompt asks for 8-12 steps and batched profiling, but isn't guaranteed to comply. Raise `--recursion-limit` for larger datasets.
- **Large datasets** — everything is held in memory (pandas); there's no sampling or chunked processing for big files.
- **Single file at a time** — no multi-table joins or multi-file ingestion.
- **No visual inspection of plots** — the LLM sees file paths for generated plots, not the pixels, so it can't visually critique a chart it just made.

**Possible next steps**

- Wire `config/config.yaml` into the nodes/tools instead of hardcoded constants.
- Implement the Docker sandbox mode for stronger isolation than RestrictedPython.
- LLM-driven feature engineering and time-series support.
- A small web UI for kicking off runs and browsing reports.

---

## License

MIT — see [LICENSE](LICENSE).

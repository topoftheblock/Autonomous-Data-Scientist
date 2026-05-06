# agent/tools/reporting.py
"""
Report‑generation tool. Called by the reporter node at the end of the pipeline.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from langchain.tools import tool

# Centralised output directory – same as visualisation tool
OUTPUT_DIR = "data/output"

@tool
def generate_report(report_data_json: str) -> str:
    """
    Create a formal Markdown report from a JSON string containing:
    {
        "file_path": "data/input/sales.csv",
        "data_summary": "{...}",
        "step_results": ["Step 1: ...", "Step 2: ..."],
        "statistical_tests": [{"test": "ttest", "p_value": 0.03, "interpretation": "..."}, ...],
        "plot_paths": ["data/output/histogram_age.png", ...],
        "correlation_matrix": { ... },
        "cleaning_summary": { ... }
    }

    The report is saved to data/output/report_{timestamp}.md
    Returns the path to the saved Markdown file.
    """
    try:
        data = json.loads(report_data_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON input."})

    # --- Build report sections ---

    sections = []
    sections.append(f"# Data Science Report\n\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    sections.append(f"**Dataset:** `{data.get('file_path', 'unknown')}`\n\n---\n")

    # 1. Data Overview
    sections.append("## 1. Data Overview\n")
    summary = data.get("data_summary", "{}")
    try:
        summary_dict = json.loads(summary) if isinstance(summary, str) else summary
        shape = summary_dict.get("shape", "?")
        cols = summary_dict.get("columns", [])
        missing = summary_dict.get("missing", {})
        sections.append(f"- **Shape:** {shape[0]} rows × {shape[1]} columns\n")
        sections.append(f"- **Columns:** {', '.join(cols)}\n")
        if missing:
            missing_str = ", ".join(f"{k}: {v}" for k, v in missing.items() if v > 0)
            sections.append(f"- **Missing values:** {missing_str if missing_str else 'None'}\n")
        sections.append("\n")
    except:
        sections.append("*(Summary not available)*\n\n")

    # 2. Data Cleaning
    sections.append("## 2. Data Cleaning\n")
    cleaning = data.get("cleaning_summary", {})
    if cleaning:
        sections.append(f"- **Original shape:** {cleaning.get('original_shape', '?')}\n")
        sections.append(f"- **Cleaned shape:** {cleaning.get('new_shape', '?')}\n")
        sections.append(f"- **Actions applied:** {json.dumps(cleaning.get('actions_applied', {}), indent=2)}\n\n")
    else:
        sections.append("*(No cleaning performed)*\n\n")

    # 3. Statistical Analysis
    sections.append("## 3. Statistical Analysis\n")
    tests = data.get("statistical_tests", [])
    if tests:
        for i, test in enumerate(tests, 1):
            sections.append(f"### Test {i}: {test.get('test', 'unknown')}\n")
            sections.append(f"- **Statistic:** {test.get('statistic', 'N/A')}\n")
            sections.append(f"- **p‑value:** {test.get('p_value', 'N/A')}\n")
            sections.append(f"- **Significant (α=0.05):** {'Yes' if test.get('significant') else 'No'}\n")
            sections.append(f"- **Interpretation:** {test.get('interpretation', '')}\n\n")
    else:
        sections.append("*(No statistical tests run)*\n\n")

    # Optional: Correlation matrix
    corr = data.get("correlation_matrix")
    if corr and "strong_correlations" in corr:
        sections.append("### Correlation Summary\n")
        strong = corr["strong_correlations"]
        if strong:
            for pair in strong:
                sections.append(f"- {pair['pair'][0]} vs {pair['pair'][1]}: r = {pair['coefficient']}\n")
        else:
            sections.append("No strong correlations (|r| > 0.5) found.\n")
        sections.append("\n")

    # 4. Visualizations
    sections.append("## 4. Visualizations\n")
    plot_paths = data.get("plot_paths", [])
    if plot_paths:
        for path in plot_paths:
            # relative path from project root for Markdown
            display_path = os.path.relpath(path, start=os.getcwd())
            # Generate a caption from the file name
            file_stem = Path(path).stem
            caption = file_stem.replace("_", " ").title()
            sections.append(f"![{caption}]({display_path})\n\n")
    else:
        sections.append("*(No plots generated)*\n\n")

    # 5. Step‑by‑Step log (collapsed)
    sections.append("## 5. Execution Log\n")
    step_results = data.get("step_results", [])
    if step_results:
        for i, step in enumerate(step_results, 1):
            sections.append(f"### Step {i}\n{step}\n\n")
    else:
        sections.append("*(No step details recorded)*\n")

    sections.append("\n---\n*Report generated automatically by the Autonomous Data Scientist Agent.*\n")

    # Write to file
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(OUTPUT_DIR, f"report_{timestamp}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(sections))

    return json.dumps({"report_path": report_path, "message": "Formal report saved."})
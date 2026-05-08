# main.py
"""
Entry point for the Autonomous Data Science Agent.
Usage:
    python main.py data/input/sales.csv
    python main.py data/input/sales.csv --output data/output/my_report.md
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv("sandbox settings/secrets.env") # or load_dotenv(".env") depending on where your file is

from agent.state import AgentState
from agent.graph import app

# ---------------------------------------------------------------
# Logging
# ---------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("datascience_agent")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Data Scientist – fully hands‑off analysis of a CSV file."
    )
    parser.add_argument(
        "file",
        type=str,
        help="Path to the CSV (or Excel) file to analyse.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/report.md",
        help="Path where the final Markdown report will be saved (default: data/output/report.md).",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=30,
        help="Maximum number of steps the agent may take (default: 30).",
    )
    args = parser.parse_args()

    # Checks
    if not Path(args.file).exists():
        logger.error(f"File not found: {args.file}")
        sys.exit(1)

    if "OPENAI_API_KEY" not in os.environ:
        logger.error("OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    # Initial state – empty except for the file path
    initial_state: AgentState = {
        "file_path": args.file,
        "data_summary": "",
        "data_parquet_path": None,
        "plan": [],
        "current_step_index": 0,
        "step_results": [],
        "error": "",
        "final_report": "",
    }

    logger.info("Starting autonomous data‑science pipeline …")
    try:
        final_state = app.invoke(initial_state, {"recursion_limit": args.recursion_limit})
    except Exception as e:
        logger.exception("Pipeline crashed with an unexpected error.")
        sys.exit(1)

    # Check for pipeline‑internal error
    if final_state.get("error"):
        logger.error(f"Pipeline finished with error: {final_state['error']}")
        sys.exit(1)

    # Success → save report
    report_path = Path(args.output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(final_state["final_report"], encoding="utf-8")
    logger.info(f"Report saved to {report_path.resolve()}")

    # Also print the report to stdout for immediate inspection
    print("\n" + "=" * 80)
    print("                     AUTONOMOUS DATA SCIENCE REPORT")
    print("=" * 80)
    print(final_state["final_report"])
    print("=" * 80)


if __name__ == "__main__":
    main()
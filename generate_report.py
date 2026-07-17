# -*- coding: utf-8 -*-
"""Generate an iterative error analysis report from attributed failed cases.

The script reads a JSON data file containing a list of failed-case records.
Each record bundles the original query, golden answers, model prediction,
source evidence, and a pre-computed error attribution result. The records are
batched and fed sequentially to the error analysis report runner, which
refines a single evolving report after each batch and saves the final report
to the configured save folder.
"""

import argparse
import asyncio
import json
import os 
from pathlib import Path
from toolkits.bench_utils import FailedQueryCase
from toolkits.error_analysis_utils import (
    ErrorAnalysisReportRunner,
    ReportGenerationConfig,
)
from toolkits.memtrace_utils import ErrorAttributionPrediction


FINAL_REPORT_FILE = "error_analysis_report.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        `argparse.Namespace`:
            Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate an iterative error analysis report from a JSON data "
            "file of attributed failed cases."
        ),
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help=(
            "Path to a JSON file containing the failed cases."
        ),
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        required=True,
        help="Folder where the final error analysis report will be written.",
    )
    parser.add_argument(
        "--target-system-overview",
        type=str,
        default=None,
        help=(
            "Optional overview of the target system being diagnosed."
            "It is either an inline string or a path to a text file "
            "whose contents will be used as the overview. The overview is "
            "embedded in the report agent's system prompt. When not provided, "
            "a generic system-agnostic overview is used."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.4",
        help="Backbone model used by the error analysis report agent.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature used by the error analysis report agent.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help=(
            "Number of failed cases per batch. After each batch, the report "
            "is updated by a single LLM call and then fed back as the "
            "current report for the next batch."
        ),
    )
    parser.add_argument(
        "--max-op-xml-tokens",
        type=int,
        default=300_000,
        help=(
            "Maximum tokens retained from one attributed operation subgraph "
            "XML. Oversized XML keeps its last tokens."
        ),
    )
    parser.add_argument(
        "--api-config-path",
        type=str,
        default=None,
        help=(
            "Path to an API config JSON with `api_keys` and `base_urls`. "
            "If not provided, the runner falls back to OpenAI-compatible "
            "environment variables."
        ),
    )
    parser.add_argument(
        "--studio-url",
        type=str,
        default=None,
        help="Optional AgentScope Studio URL for live inspection.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="error_analysis_report",
        help="AgentScope Studio project name.",
    )
    return parser.parse_args()


def load_cases(
    data_path: str | Path,
) -> tuple[
    list[FailedQueryCase],
    list[ErrorAttributionPrediction],
    list[str],
]:
    """Load failed cases, attributions, and graph paths from the data file.

    Args:
        data_path (`str | Path`):
            Path to the JSON data file.

    Returns:
        `tuple[list[FailedQueryCase], list[ErrorAttributionPrediction], list[str]]`:
            Failed query cases, attributed error predictions, and per-record
            graph paths aligned with the records in the data file.
    """
    with open(data_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(
            f"Data file '{data_path}' must contain a JSON list of records."
        )

    failed_cases = []
    error_predictions = []
    graph_paths = []

    for record in data:
        if not isinstance(record, dict):
            raise ValueError(
                f"Each record in '{data_path}' must be a JSON object."
            )

        golden_answers = record["golden_answers"]
        metadata = {
            "golden_answers": golden_answers,
            "golden_answers_id": record["golden_answers_id"],
            "prediction_id": record["prediction_id"],
        }
        failed_cases.append(
            FailedQueryCase(
                query_full_node_id=record["query_id"],
                query=record["query"],
                golden_answer=", ".join(golden_answers),
                prediction=record["prediction"],
                source_evidence_full_node_ids=record["source_evidence_ids"],
                metadata=metadata,
            )
        )
        error_predictions.append(
            ErrorAttributionPrediction.model_validate(
                record["error_attribution_result"],
            )
        )
        graph_paths.append(record["graph_path"])

    return failed_cases, error_predictions, graph_paths


async def run(args: argparse.Namespace) -> None:
    """Run the error analysis report generation pipeline.

    Args:
        args (`argparse.Namespace`):
            Parsed command-line arguments.
    """
    failed_cases, error_predictions, graph_paths = load_cases(
        data_path=args.data_path,
    )
    config = ReportGenerationConfig(
        model=args.model,
        temperature=args.temperature,
        stream=True,
        batch_size=args.batch_size,
        max_op_xml_tokens=args.max_op_xml_tokens,
        api_config_path=args.api_config_path,
        studio_url=args.studio_url,
        project=args.project,
    )
    runner = ErrorAnalysisReportRunner(config=config)

    target_system_overview = args.target_system_overview
    if os.path.isfile(target_system_overview):
        with open(target_system_overview, "r", encoding="utf-8") as file:
            target_system_overview = file.read()

    report, costs = await runner.arun(
        failed_cases=failed_cases,
        error_predictions=error_predictions,
        target_system_overview=target_system_overview,
        graph_paths=graph_paths,
        save_folder=args.save_folder,
    )

    print() 
    print("Final error analysis report:")
    print(report)
    print("-" * 50)
    print() 
    print("-" * 50)
    for key, value in costs.items():
        print(f"{key}: {value}")
    print("-" * 50)
    

def main() -> None:
    """Run the command-line entry point."""
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()

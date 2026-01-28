#!/usr/bin/env python
"""Run the news ingestor pipeline (drain_pubsub through topic_jobs by default)."""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Add scripts dir so sibling scripts can be imported
sys.path.insert(0, str(Path(__file__).parent))

# Load .env before importing anything else
from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Run the ingestion pipeline")
    parser.add_argument("--start-step", type=str, default="drain_pubsub",
                        help="First pipeline step to execute (default: drain_pubsub)")
    parser.add_argument("--end-step", type=str, default="topic_jobs",
                        help="Final pipeline step to execute (default: topic_jobs)")
    parser.add_argument("--max-messages", type=int, default=None,
                        help="Maximum Pub/Sub messages to drain")
    parser.add_argument("--max-stance-jobs", type=int, default=None,
                        help="Maximum stance jobs to process")
    parser.add_argument("--max-clusters", type=int, default=None,
                        help="Maximum clusters to process")
    parser.add_argument("--max-topic-clusters", type=int, default=None,
                        help="Maximum clusters to assign topics to")
    parser.add_argument("--bypass-stance-llm", action="store_true",
                        help="Bypass LLM calls for stance jobs")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    # Import here to ensure path is set up
    from news_pipeline.ingestor.run import main as run_main

    # Construct sys.argv for the underlying main function
    sys.argv = ["run_ingestor.py"]
    if args.start_step:
        sys.argv.extend(["--start-step", args.start_step])
    if args.end_step:
        sys.argv.extend(["--end-step", args.end_step])
    if args.max_messages:
        sys.argv.extend(["--max-messages", str(args.max_messages)])
    if args.max_stance_jobs:
        sys.argv.extend(["--max-stance-jobs", str(args.max_stance_jobs)])
    if args.max_clusters:
        sys.argv.extend(["--max-clusters", str(args.max_clusters)])
    if args.max_topic_clusters:
        sys.argv.extend(["--max-topic-clusters", str(args.max_topic_clusters)])
    if args.bypass_stance_llm:
        sys.argv.append("--bypass-stance-llm")
    if args.verbose:
        sys.argv.append("--verbose")

    run_main()

    # Automatically populate fallback articles after ingestion
    from populate_fallback import main as populate_fallback_main
    print("\n--- Running fallback article population ---")
    # Reset sys.argv to avoid passing unrecognized arguments
    sys.argv = ["populate_fallback.py"]
    populate_fallback_main()


if __name__ == "__main__":
    main()

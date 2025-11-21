"""Main entry point for the AI publication analysis pipeline."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from openai import OpenAI
import polars as pl

from src.analyse import analyze_abstract, compute_rating
from src.data import load_abstracts, load_checkpoint, load_examples, save_checkpoint

# Define project paths
PROJECT_ROOT = Path(__file__).parent.absolute()
INPUT_DIR = PROJECT_ROOT / "in"
OUTPUT_DIR = PROJECT_ROOT / "out"

# Define file paths
EXAMPLES_FILE = INPUT_DIR / "exemples.json"
ABSTRACTS_FILE = INPUT_DIR / "abstracts.csv"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"

# Configuration constants
MODEL_NAME = "gpt-5-nano-2025-08-07"
CHECKPOINT_INTERVAL = 1

# Initialize logger (configuration happens in main())
logger = logging.getLogger(__name__)


def main():
    """Run the analysis pipeline."""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(OUTPUT_DIR / "analysis.log"), logging.StreamHandler()],
    )

    # Validate API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is not set")
        logger.error("Please set it with: export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)

    # Initialize OpenAI client
    client = OpenAI()

    # Find current timestamp
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"evaluated_by_{MODEL_NAME}-{now}.csv"

    # Load data
    logger.info("Loading examples and abstracts...")
    examples = load_examples(EXAMPLES_FILE)
    abstracts = load_abstracts(ABSTRACTS_FILE)

    # Load checkpoint if exists
    results = load_checkpoint(CHECKPOINT_FILE)
    processed_dois = {r["doi"] for r in results}

    # Set logger
    logger.info(f"Starting analysis with model {MODEL_NAME}")
    counter = len(results)

    # Run analysis on all abstracts
    for row in abstracts.iter_rows(named=True):
        # Skip if already processed
        if row["DOI"] in processed_dois:
            continue

        # Log progress
        logger.info(f"Analyzing DOI # {counter + 1}: {row['DOI']}")
        counter += 1

        # Get analysis
        try:
            result = analyze_abstract(
                row["Abstract.Note"], examples, model=MODEL_NAME, client=client
            )

            # Create a new dict with rating
            result_dict = {
                "doi": row["DOI"],
                "abstract": row["Abstract.Note"],
                **result,
                "rating": compute_rating(result),
            }

            # Append to results
            results.append(result_dict)

        except Exception as e:
            logger.error(f"Failed to analyze DOI {row['DOI']}: {e}")
            # Save checkpoint to preserve progress before error
            save_checkpoint(CHECKPOINT_FILE, results)
            logger.info(f"Checkpoint saved after error at {counter} abstracts")
            continue

        # Save checkpoint every 10 abstracts
        if counter % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(CHECKPOINT_FILE, results)
            logger.info(f"Checkpoint saved at {counter} abstracts")

    # Save final results
    pl.DataFrame(results).write_csv(output_file)

    # Remove checkpoint file after successful completion
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("Checkpoint file removed after successful completion")

    logger.info(f"Analysis complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()

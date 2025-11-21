"""Data loading and persistence functions for the analysis pipeline."""

import json
import logging
import sys
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def load_examples(examples_file: Path) -> list:
    """
    Load and validate example abstracts from JSON file.

    Args:
        examples_file: Path to the examples JSON file

    Returns:
        List of example abstracts with analysis fields
    """
    if not examples_file.exists():
        logger.error(f"Examples file not found at {examples_file}")
        sys.exit(1)

    try:
        with open(examples_file, "r") as file:
            examples = json.load(file)

        # Validate JSON structure
        if not isinstance(examples, list):
            logger.error("Examples file must contain a JSON array")
            sys.exit(1)

        required_fields = [
            "abstract",
            "participatory_method",
            "participatory_method_rationale",
            "green_infrastructure_intervention",
            "green_infrastructure_rationale",
        ]
        for i, example in enumerate(examples):
            for field in required_fields:
                if field not in example:
                    logger.error(f"Example {i} missing required field '{field}'")
                    sys.exit(1)

        return examples

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in examples file: {e}")
        sys.exit(1)


def load_abstracts(abstracts_file: Path) -> pl.DataFrame:
    """
    Load and validate abstracts from CSV file.

    Args:
        abstracts_file: Path to the abstracts CSV file

    Returns:
        Polars DataFrame containing abstracts
    """
    if not abstracts_file.exists():
        logger.error(f"Abstracts file not found at {abstracts_file}")
        sys.exit(1)

    try:
        abstracts = pl.read_csv(abstracts_file)

        # Validate required columns
        required_columns = ["DOI", "Abstract.Note"]
        for col in required_columns:
            if col not in abstracts.columns:
                logger.error(f"CSV missing required column '{col}'")
                sys.exit(1)

        return abstracts

    except Exception as e:
        logger.error(f"Error reading abstracts CSV: {e}")
        sys.exit(1)


def load_checkpoint(checkpoint_file: Path) -> list:
    """
    Load checkpoint if it exists.

    Args:
        checkpoint_file: Path to the checkpoint file

    Returns:
        List of previously processed results, or empty list if no checkpoint exists
    """
    if checkpoint_file.exists():
        logger.info(f"Loading checkpoint from {checkpoint_file}")
        with open(checkpoint_file, "r") as f:
            results = json.load(f)
        logger.info(f"Resuming from {len(results)} processed abstracts")
        return results
    return []


def save_checkpoint(checkpoint_file: Path, results: list) -> None:
    """
    Save checkpoint to disk.

    Args:
        checkpoint_file: Path to the checkpoint file
        results: List of results to save
    """
    with open(checkpoint_file, "w") as f:
        json.dump(results, f, indent=2)

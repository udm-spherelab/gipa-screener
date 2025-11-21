"""Test to verify the full analysis workflow returns expected output format."""

import json
import logging
import os
import sys
from pathlib import Path

from openai import OpenAI
import polars as pl

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from src.analyse import analyze_abstract, compute_rating

# Configure logging for test
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"


def load_examples(examples_file: Path) -> list:
    """Load and validate example abstracts from JSON file."""
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
    """Load and validate abstracts from CSV file."""
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


def test_full_workflow():
    """Test the full workflow with first abstract only."""
    logger.info("="*60)
    logger.info("TESTING FULL WORKFLOW WITH FIRST ABSTRACT")
    logger.info("="*60)

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is not set")
        sys.exit(1)

    # Define file paths (same as main.py)
    examples_file = DATA_DIR / "exemples.json"
    abstracts_file = DATA_DIR / "not_evaluated.csv"

    # Load data (same as main.py)
    logger.info("Loading examples and abstracts...")
    examples = load_examples(examples_file)
    abstracts = load_abstracts(abstracts_file)

    logger.info(f"Loaded {len(examples)} examples")
    logger.info(f"Loaded {len(abstracts)} abstracts (will test with first one only)")

    # Initialize OpenAI client (same as main.py)
    client = OpenAI()
    model_name = "gpt-5-nano-2025-08-07"

    logger.info(f"Using model: {model_name}")

    # Get first abstract only
    first_row = abstracts.row(0, named=True)

    logger.info("\n" + "-"*60)
    logger.info("PROCESSING FIRST ABSTRACT")
    logger.info("-"*60)
    logger.info(f"DOI: {first_row['DOI']}")
    logger.info(f"Abstract (first 150 chars): {first_row['Abstract.Note'][:150]}...")

    # Run the same workflow as main.py
    logger.info(f"\nAnalyzing abstract...")
    result = analyze_abstract(
        first_row["Abstract.Note"], examples, model=model_name, client=client
    )

    # Create result_dict (same as main.py)
    result_dict = {
        "doi": first_row["DOI"],
        "abstract": first_row["Abstract.Note"],
        **result,
        "rating": compute_rating(result),
    }

    # Validate output format
    logger.info("\n" + "="*60)
    logger.info("VALIDATING OUTPUT FORMAT")
    logger.info("="*60)

    all_checks_passed = True

    # Check all expected keys in result_dict
    expected_keys = [
        "doi",
        "abstract",
        "participatory_method",
        "participatory_method_rationale",
        "green_infrastructure_intervention",
        "green_infrastructure_rationale",
        "rating",
    ]

    logger.info("\nChecking required fields:")
    for key in expected_keys:
        if key in result_dict:
            logger.info(f"  ✓ '{key}' present")
        else:
            logger.error(f"  ✗ '{key}' MISSING")
            all_checks_passed = False

    # Validate field types
    logger.info("\nValidating field types:")

    if isinstance(result_dict.get("doi"), str):
        logger.info(f"  ✓ doi: str")
    else:
        logger.error(f"  ✗ doi has wrong type: {type(result_dict.get('doi'))}")
        all_checks_passed = False

    if isinstance(result_dict.get("abstract"), str):
        logger.info(f"  ✓ abstract: str")
    else:
        logger.error(f"  ✗ abstract has wrong type: {type(result_dict.get('abstract'))}")
        all_checks_passed = False

    if isinstance(result_dict.get("participatory_method"), (bool, type(None))):
        logger.info(f"  ✓ participatory_method: {type(result_dict['participatory_method']).__name__}")
    else:
        logger.error(f"  ✗ participatory_method has wrong type: {type(result_dict.get('participatory_method'))}")
        all_checks_passed = False

    if isinstance(result_dict.get("participatory_method_rationale"), str):
        logger.info(f"  ✓ participatory_method_rationale: str")
    else:
        logger.error(f"  ✗ participatory_method_rationale has wrong type: {type(result_dict.get('participatory_method_rationale'))}")
        all_checks_passed = False

    if isinstance(result_dict.get("green_infrastructure_intervention"), (bool, type(None))):
        logger.info(f"  ✓ green_infrastructure_intervention: {type(result_dict['green_infrastructure_intervention']).__name__}")
    else:
        logger.error(f"  ✗ green_infrastructure_intervention has wrong type: {type(result_dict.get('green_infrastructure_intervention'))}")
        all_checks_passed = False

    if isinstance(result_dict.get("green_infrastructure_rationale"), str):
        logger.info(f"  ✓ green_infrastructure_rationale: str")
    else:
        logger.error(f"  ✗ green_infrastructure_rationale has wrong type: {type(result_dict.get('green_infrastructure_rationale'))}")
        all_checks_passed = False

    if isinstance(result_dict.get("rating"), int) and result_dict["rating"] in [1, 2, 3]:
        logger.info(f"  ✓ rating: int (value={result_dict['rating']})")
    else:
        logger.error(f"  ✗ rating has wrong type or value: {result_dict.get('rating')}")
        all_checks_passed = False

    # Display full result
    logger.info("\n" + "-"*60)
    logger.info("FULL RESULT FROM WORKFLOW")
    logger.info("-"*60)

    # Pretty print without abstract (too long)
    display_result = {k: v for k, v in result_dict.items() if k != "abstract"}
    display_result["abstract"] = result_dict["abstract"][:100] + "..."

    logger.info(json.dumps(display_result, indent=2))

    # Final summary
    logger.info("\n" + "="*60)
    if all_checks_passed:
        logger.info("✓ TEST PASSED - Full workflow returned correct format")
        logger.info("="*60)
        return 0
    else:
        logger.error("✗ TEST FAILED - Format validation issues detected")
        logger.info("="*60)
        return 1


if __name__ == "__main__":
    exit_code = test_full_workflow()
    sys.exit(exit_code)

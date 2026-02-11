"""Core analysis functions for evaluating research abstracts."""

from openai import OpenAI
from pydantic import BaseModel

from src.prompt import generate_messages


class Output(BaseModel):
    """Schema for analysis output."""

    participatory_method: bool | None
    participatory_method_rationale: str
    green_infrastructure_intervention: bool | None
    green_infrastructure_rationale: str


def analyze_abstract(
    abstract: str, examples: list, model: str, client: OpenAI
) -> dict[str, any]:
    """
    Analyze a research abstract using OpenAI's API.

    Args:
        abstract: The abstract text to analyze
        examples: List of example abstracts for few-shot learning
        model: OpenAI model name to use
        client: OpenAI client instance

    Returns:
        Dictionary with analysis results including participatory_method,
        participatory_method_rationale, green_infrastructure_intervention,
        and green_infrastructure_rationale
    """
    messages = generate_messages(abstract, examples)
    response = client.responses.parse(
        model=model, input=messages, text_format=Output, temperature=0, seed=42
    )
    return response.output_parsed.model_dump()


def compute_rating(result: dict[str, any]) -> int:
    """
    Compute rating based on analysis results.

    Args:
        result: Dictionary with participatory_method and green_infrastructure_intervention fields

    Returns:
        Rating: 3 if both true, 1 if at least one false, 2 otherwise
    """
    if result["participatory_method"] and result["green_infrastructure_intervention"]:
        return 3
    elif (
        not result["green_infrastructure_intervention"]
        or not result["participatory_method"]
    ):
        return 1
    return 2

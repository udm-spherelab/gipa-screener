"""Core analysis functions for evaluating research abstracts."""

from openai import OpenAI
from pydantic import BaseModel

from src.prompt import generate_messages


class Output(BaseModel):
    """Schema for analysis output."""

    participatory_method: bool | None
    participatory_method_rationale: str
    participation_degree: str | None
    participant_types: str | None 
    green_infrastructure_intervention: bool | None
    green_infrastructure_rationale: str
    setting: str | None

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
        Dictionary with analysis results including:
        - participatory_method: bool or None - Whether participatory methods are used
        - participatory_method_rationale: str - Justification for participatory_method
        - participation_degree: str or None - Level of participation
        - participant_types: str or None - Types of participants involved
        - green_infrastructure_intervention: bool or None - Whether green infrastructure intervention is present
        - green_infrastructure_rationale: str - Justification for green_infrastructure_intervention
        - setting: str or None - Study setting (urban/rural/mixed)
    """
    messages = generate_messages(abstract, examples)
    response = client.responses.parse(
        model=model, input=messages, text_format=Output, reasoning={"effort": "high"},
    )
    return response.output_parsed.model_dump()

def compute_rating(result: dict[str, any]) -> int:
    """
    Compute rating based on analysis results, reproducing exactly the
    definitions in Table 4 of the thesis (title/abstract screening):
        3 - both criteria TRUE + urban or peri-urban context
        2 - at least one criterion is None/unclear, OR both criteria TRUE
            but mixed/undetermined context (needs full text review)
        1 - at least one criterion FALSE, OR both TRUE but rural context

    Args:
         result: Dictionary with participatory_method, green_infrastructure_intervention, 
                setting

    Returns:
        Rating: 1, 2, or 3 (see above)
    """
    pm = result["participatory_method"]
    gi = result["green_infrastructure_intervention"]
    setting = result.get("setting")

    # Case 1: At least one unclear criterion -> Rating 2
    if pm is None or gi is None:
        return 2

    # Case 2: At least one criterion FALSE (includes forced review documents,
    # where the prompt sets both pm and gi to false) -> Rating 1
    if not (pm and gi):
        return 1

    # Case 3: Both criteria TRUE -> rating depends on setting
    if setting in ("urban", "peri_urban"):
        return 3
    if setting == "rural":
        return 1

    # setting == "mixed", or None/undetermined despite pm and gi both TRUE
    return 2

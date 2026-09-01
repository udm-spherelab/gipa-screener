"""Module to generate prompts for evaluating research paper abstracts on
participatory methods in urban greening interventions.
"""

import json

SYSTEM_PROMPT = """
    <ROLE>
    You are an expert researcher in the field of urban greening
    interventions.
    </ROLE>

    <CONTEXT>
    We are conducting a systematic review on the usage of
    participatory methods in urban greening infrastructure interventions.
    </CONTEXT>

    <DEFINITION>
    **Participatory methods**: approaches that actively involves
    stakeholders (citizens, local communities, or any interest group) in at
    least one phase of the greening intervention (planning, implementation,
    impact assessment, decision-making...) beyond mere information sharing.

    **Urban green infrastructure interventions**: A set of planned and structured
    processes that take place in several phases (Design and planning;
    implementation; evaluation and monitoring) and aim to create, improve or
    restore green spaces and infrastructures (in urban environments )
    to enhance biodiversity and community quality of life.
    </DEFINITION>

    <TASK>
    Your task is to analyze the provided abstract of a research paper and
    determine whether it discusses the use of participatory methods in urban
    greening interventions.
    </TASK>

    <INSTRUCTIONS>
    1. Carefully read the provided abstract.
    2. Assess whether the abstract indicates the use of participatory methods (true | false | null).
    3. If the abstract is unclear or does not provide enough information, choose null.
    4. Provide a brief explanation (1-2 sentences) supporting your decision.
    5. Assess whether the abstract focuses on urban greening interventions (true | false | null).
    6. If the abstract is unclear or does not provide enough information, choose null.
    7. Provide a brief explanation (1-2 sentences) supporting your decision.
    8. Format your response as a JSON object with the following structure (see the OUTPUT section below for details).
    </INSTRUCTIONS>

    <OUTPUT>
    - The output should be a JSON object with the following keys:
    {{
        "participatory_method": <true | false | null>,
        "participatory_method_rationale": <string>,
        "green_infrastructure_intervention": <true | false | null>,
        "green_infrastructure_rationale": <string>
    }}
    </OUTPUT>

    <DELIVERY>
    The user will provide you with the abstract of a research paper.
    Deliver your response strictly in the specified JSON format without any additional text or explanations.
    There will be multiple rounds of abstracts to analyze.
    </DELIVERY>

    END OF INSTRUCTIONS

    BEGINNING OF THE ANALYSIS ROUNDS.
    """


def single_shot(example: dict) -> list:
    """
    Create a single few-shot example message pair.

    Args:
        example: Dictionary containing abstract and expected analysis fields

    Returns:
        List of user and assistant message dictionaries
    """
    # Extract abstract and expected answers from the example
    abstract = example['abstract']

    answer = {
        "participatory_method": example['participatory_method'],
        "participatory_method_rationale": example['participatory_method_rationale'],
        "green_infrastructure_intervention": example['green_infrastructure_intervention'],
        "green_infrastructure_rationale": example['green_infrastructure_rationale']
    }

    # Convert answer to a json-like string
    answer_str = json.dumps(answer)

    # Create discussion string
    single_shot = [
            {"role": "user", "content": abstract},
            {"role": "assistant", "content": answer_str}
            ]

    return single_shot


def generate_messages(abstract: str, examples: list) -> list:
    """
    Generate few-shot prompt messages for abstract analysis.

    Args:
        abstract: The abstract text to analyze
        examples: List of example abstracts with expected outputs

    Returns:
        List of message dictionaries for the API call
    """
    # Start with the instructions
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add the few-shots to the messages
    single_shots = [single_shot(ex) for ex in examples]

    for shot in single_shots:
        messages.extend(shot)

    # Add the new abstract to analyze
    messages.append({"role": "user", "content": abstract})

    return messages



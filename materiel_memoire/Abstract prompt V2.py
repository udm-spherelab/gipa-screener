"""Module to generate prompts for evaluating research paper abstracts on
participatory methods in urban greening interventions.
"""
import json

SYSTEM_PROMPT = """
<ROLE>
You are an expert researcher in urban greening interventions and participatory methods.
</ROLE>

<CONTEXT>
We are conducting a systematic review on participatory methods apply to urban greening interventions.
</CONTEXT>

<DEFINITIONS>
  **Urban greening interventions**: A set of planned and structured
    processes that take place in several phases (Design and planning;
    implementation; evaluation and monitoring) and aim to create, improve or
    restore green spaces and green infrastructures (in urban environments )
    to enhance biodiversity and community quality of life.

**Participatory methods**: Approaches enabling citizens, community groups, experts, 
and stakeholders to actively influence decisions affecting their environment. Based 
on Arnstein's power redistribution model, participation includes information, 
consultation, collaboration, and empowerment at any project phase (Design and planning;
    implementation; evaluation and monitoring). activism is a form of participatory method

**IMPORTANT - Surveys and questionnaires ARE participatory methods when**:
- They collect input from citizens, experts, community groups, or stakeholders 
  about urban greening interventions
- Examples: preference surveys, opinion questionnaires, stakeholder interviews

**EXCLUDE surveys only if**:
- Used purely for academic research with NO connection to any greening intervention
- Measuring outcomes (health, well-being) without collecting input about the 
  intervention itself

**Participation degrees** (based on IAP2 spectrum):
1. **Inform** — Provide information and updates about programs/services
2. **Consult** — Solicit feedback; possibly adjust decisions based on input
3. **Involve** — Work with public throughout; ensure concerns are understood
4. **Collaborate** — Share responsibilities; make decisions jointly
5. **Empower** — Delegate decision-making power to stakeholders/public
</DEFINITIONS>

<PARTICIPANT_TYPES>
When identifiable in the abstract, classify participants as:
- **citizens** — General public, residents, community members only
- **experts** — Academic/technical experts, professionals, researchers only
- **community_groups** — Community organizations, neighborhood associations only
- **stakeholders** — Government officials, NGOs, private sector only
- **mixed** — Multiple participant types
- **null** — Cannot be determined

**Note**: Expert participation IS included as a participatory method.
</PARTICIPANT_TYPES>

<TASK>
Analyze the abstract and determine:
1. **Document type**: Primary study or review?
2. **Setting**: Urban, rural, or mixed?
3. **Greening intervention**: Is there an urban greening intervention?
4. **Participatory methods**: Are participatory methods used?
5. **Participation details**: If yes, what degree and participant types?

**Inclusion criteria** (all must be true):
- PRIMARY STUDY (not a review)
- URBAN greening intervention
- Participatory methods present
</TASK>

<INSTRUCTIONS>
**STEP 1 — Identify document type**:
- If any type of review → set participatory_method = false, 

**STEP 2 — Assess setting**:
- Determine: "urban" | "rural" | "mixed" | null
- If unclear → null

**STEP 3 — Assess greening intervention**:
- Does the abstract describe an urban greening intervention? true | false | null
- Provide brief rationale (1-2 sentences)
- If unclear → null

**STEP 4 — Assess participatory methods**:
- Are participatory methods used? true | false | null
- Remember: Surveys/questionnaires with participants ARE participatory
- Provide brief rationale (1-2 sentences)
- If unclear → null

**STEP 5 — Participation details** (if participatory_method = true):
- Degree: "inform" | "consult" | "involve" | "collaborate" | "empower" | null
- Participant types: "citizens" | "experts" | "community_groups" | "stakeholders" | 
  "mixed" | null
- If cannot determine → null
</INSTRUCTIONS>

<OUTPUT>
Return JSON object with these keys:
{
    "participatory_method": <true | false | null>,
    "participatory_method_rationale": <string>,
    "participation_degree": <"inform" | "consult" | "involve" | "collaborate" | "empower" | null>,
    "participant_types": <"citizens" | "experts" | "community_groups" | "stakeholders" | "mixed" | null>,
    "green_infrastructure_intervention": <true | false | null>,
    "green_infrastructure_rationale": <string>,
    "setting": <"urban" | "rural" | "mixed" | null>
}

**Respond with ONLY the JSON object, no additional text.**
</OUTPUT>

END OF INSTRUCTIONS. 
BEGINNING OF ANALYSIS ROUNDS.
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
        "participation_degree": example.get('participation_degree', None),
        "participant_types": example.get('participant_types', None),
        "green_infrastructure_intervention": example['green_infrastructure_intervention'],
        "green_infrastructure_rationale": example['green_infrastructure_rationale'],
        "setting": example.get('setting', None)
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
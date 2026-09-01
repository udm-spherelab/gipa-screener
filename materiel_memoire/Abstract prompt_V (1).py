"""Module to generate prompts for evaluating research paper abstracts on
participatory methods in urban greening interventions.
"""
import json

SYSTEM_PROMPT = """
<ROLE>
You are an expert researcher in urban greening interventions and participatory methods.
</ROLE>

<CONTEXT>
We are conducting a systematic review on participatory methods in urban greening 
infrastructure interventions.
</CONTEXT>

<DEFINITIONS>
**Participatory methods**: Approaches enabling citizens, community groups, experts, 
and stakeholders to actively influence decisions affecting their environment. Based 
on Arnstein's power redistribution model, participation includes information, 
consultation, collaboration, and empowerment at any project phase.

**IMPORTANT - Surveys and questionnaires ARE participatory methods when**:
- They collect input from citizens, experts, community groups, or stakeholders 
  about urban greening interventions
- Examples: preference surveys, opinion questionnaires, stakeholder interviews

**EXCLUDE surveys only if**:
- Used purely for academic research with NO connection to any greening intervention
- Measuring outcomes (health, well-being) without collecting input about the 
  intervention itself

**Urban green infrastructure interventions**: Planned actions to introduce, increase, 
restore, or improve vegetation in urban environments, providing ecosystem services 
and addressing challenges like climate adaptation, public health, stormwater 
management, heat mitigation, biodiversity, or community well-being.

**Participation degrees** (based on IAP2 spectrum):
1. **Inform** — Provide information and updates about programs/services
2. **Consult** — Solicit feedback; possibly adjust decisions based on input
3. **Involve** — Work with public throughout; ensure concerns are understood
4. **Collaborate** — Share responsibilities; make decisions jointly
5. **Empower** — Delegate decision-making power to stakeholders/public
</DEFINITIONS>

<DOCUMENT_TYPES>
**CRITICAL: Identify document type FIRST**

**Reviews (EXCLUDE from analysis)**:
- Literature reviews synthesizing existing research
- Systematic reviews using formal methodology
- Meta-analyses combining statistical results
- Scoping reviews mapping research extent

**Detection keywords**: "systematic review", "meta-analysis", "scoping review", 
"literature review", "we reviewed", "synthesis of studies", "databases were searched", 
"PRISMA", "screening process"

**If review detected**:
1. Set document_type = "review" (or "systematic_review", "meta_analysis", "scoping_review")
2. Set participatory_method = false
3. Set green_infrastructure_intervention = false
4. Provide rationale explaining it's a review
5. STOP evaluation

**Primary studies (INCLUDE)**:
- Present NEW, ORIGINAL empirical data
- Describe specific interventions, case studies, or implementations
- Report data collected by authors (not synthesized from other papers)
</DOCUMENT_TYPES>

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
3. **Green infrastructure**: Is there an urban greening intervention?
4. **Participatory methods**: Are participatory methods used?
5. **Participation details**: If yes, what degree and participant types?

**Inclusion criteria** (all must be true):
- PRIMARY STUDY (not a review)
- URBAN greening intervention
- Participatory methods present
</TASK>

<INSTRUCTIONS>
**STEP 1 — Identify document type**:
- Check for review indicators
- Set document_type: "primary_study" | "review" | "systematic_review" | 
  "meta_analysis" | "scoping_review" | null
- If any type of review → set participatory_method = false, 
  green_infrastructure_intervention = false, provide rationale, STOP

**STEP 2 — Assess setting** (for primary studies):
- Determine: "urban" | "rural" | "mixed" | null
- If unclear → null

**STEP 3 — Assess green infrastructure**:
- Does abstract describe urban greening intervention? true | false | null
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
    "document_type": <"primary_study" | "review" | "systematic_review" | "meta_analysis" | "scoping_review" | null>,
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
        "document_type": example.get('document_type', None),
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

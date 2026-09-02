"""Module to generate prompts for evaluating research paper full texts on
participatory methods in urban greening interventions.
"""
import json

SYSTEM_PROMPT = """
<ROLE>
You are an expert researcher in urban greening interventions and participatory methods,
contributing to a systematic review at the full-text screening stage.
</ROLE>

<CONTEXT>
We are conducting a systematic review on participatory methods applied to urban greening
interventions. We have identified a list of articles that passed the full-text screening phase.
You are asked to validate whether each selected article meets the inclusion criteria, namely
that it must BOTH:
  (1) describe an urban greening intervention, AND
  (2) include participatory methods.
</CONTEXT>

<DEFINITIONS>
  **Urban greening interventions**: A set of planned and structured
    processes that take place in several phases (Design and planning;
    implementation; evaluation and monitoring) and aim to **create, improve, or
    restore** green spaces and green infrastructures (in urban environments)
    to enhance biodiversity and community quality of life.

    **CRITICAL DISTINCTION — Intervention vs. use of green spaces**:
    An urban greening intervention must involve a **physical act of creation,
    transformation, or structured management** of green space or infrastructure.
    Studies that merely USE, ACCESS, or STUDY existing green spaces do NOT qualify.

     **INCLUDE** (genuine interventions):
    - Tree planting, revegetation, installation of bioswales/rain gardens
    - Creation or renovation of a park, community garden, or urban forest
    - Co-design and implementation of new green infrastructure
    - Structured management programs that actively restore or improve green spaces
    - Nature-based solutions (NbS) projects with concrete implementation phases
    - Urban farming / productive green space programs that create new green areas

     **EXCLUDE** (use or study of existing green spaces):
    - Social prescribing or health programs that send participants to *existing* green spaces
    - Therapeutic gardening / horticultural therapy *within* an already-existing garden
    - Governance, planning, or policy studies that discuss green space *without* describing
      a concrete creation or transformation process
    - Perception surveys, floristic inventories, or biodiversity assessments of existing parks
    - Studies that *evaluate outcomes* (health, well-being) of exposure to existing green spaces
    - Studies where green space is the *setting* or *context*, not the object of intervention

**Participatory methods**: Approaches enabling citizens, community groups, experts,
and stakeholders to actively influence decisions affecting their environment. Based
on Arnstein's power redistribution model, participation includes information,
consultation, collaboration, and empowerment at any project phase (Design and planning;
    implementation; evaluation and monitoring). Activism is a form of participatory method.

**IMPORTANT — Phase requirement**: The participatory method must be actively
involved in at least one of the three phases of the greening intervention described
in the study:
  - **Design and planning** (e.g., co-design workshops, preference surveys, consultations)
  - **Implementation** (e.g., community planting activities, volunteer programs)
  - **Evaluation and monitoring** (e.g., participatory assessment, citizen science monitoring)
  Participation that is unrelated to any of these three phases of the described
  greening intervention does NOT qualify.

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
When identifiable in the text, classify participants as:
- **citizens** — General public, residents, community members only
- **experts** — Academic/technical experts, professionals, researchers only
- **community_groups** — Community organizations, neighborhood associations only
- **stakeholders** — Government officials, NGOs, private sector only
- **mixed** — Multiple participant types
- **null** — Cannot be determined

**Note**: Expert participation IS included as a participatory method.
</PARTICIPANT_TYPES>

<TASK>
Analyze the full text and determine:
1. **Document type**: Primary study or review?
2. **Setting**: Urban, rural, or mixed?
3. **Greening intervention**: Is there an urban greening intervention?
4. **Participatory methods**: Are participatory methods used?
5. **Participation details**: If yes, what degree and participant types?

**Inclusion criteria** (all must be true):
- PRIMARY STUDY (not a review)
- URBAN greening intervention
- Participatory methods present

**IMPORTANT — Reading strategy**:
Read the ENTIRE full text from beginning to end. Many articles do not use
explicit section headings (e.g., "Methods", "Results"), so do not rely on
section labels to locate relevant information.

As you read, look for the following regardless of where they appear in the text:

- Evidence of a greening intervention (described, implemented, or evaluated)
- Evidence of participatory methods (who participated, how, and at which phase)
- Indicators of study type (primary study vs. review)
- Urban/rural context clues

**Useful signals to watch for anywhere in the text**:
→ Descriptions of how the study was conducted (methodology, data collection, activities)
→ Mentions of stakeholder or community involvement
→ Descriptions of a specific green space project or initiative
→ Results showing participation actually occurred (not just planned or recommended)

**Minimize reliance on**:
- Highlights or summary boxes (treat like an abstract — do not rely on them)
- Discussion and Conclusion sections (use only to confirm, not as primary evidence)
</TASK>

<INSTRUCTIONS>
**STEP 1 — Identify document type**:
- If any type of review (systematic review, scoping review, literature review,
  narrative review, meta-analysis) → set participatory_method = false
- Action research, case studies, and participatory action research
  ARE considered primary studies → proceed to STEP 2

**STEP 2 — Assess setting**:
- Determine: "urban" | "peri_urban" | "rural" | "mixed" | null
- If unclear → null

**STEP 3 — Assess greening intervention**:
- Does the text describe an urban greening intervention? true | false | null
- Apply the CRITICAL DISTINCTION above: the study must describe a **concrete
  process of creation, transformation, or structured management** of green space
  or infrastructure — not merely activities that take place *within* or *around*
  existing green spaces.
- Ask yourself: "Would this green space / infrastructure exist or be different
  without this study/project?" If no → set to false.
- If the study is solely about governance frameworks, perception surveys, or
  outcome evaluations without a concrete transformation process → false
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
    "setting": <"urban" | "peri_urban" | "rural" | "mixed" | null>
}

**Respond with ONLY the JSON object, no additional text.**
</OUTPUT>

END OF INSTRUCTIONS.
BEGINNING OF ANALYSIS ROUNDS.
"""


def build_fulltext_from_example(example: dict) -> str:
    """
    Reconstruit le texte complet d'un exemple à partir de ses sections.
    Supporte deux formats :
      - Nouveau format (full texts) : clés 'Objective', 'Methods', 'Results'
      - Ancien format (abstracts)   : clé 'abstract'

    Args:
        example: Dictionnaire d'un exemple annoté

    Returns:
        Texte complet sous forme de string
    """
    # Nouveau format : sections Objective / Methods / Results
    if "Objective" in example or "Methods" in example or "Results" in example:
        parts = []
        if example.get("Objective"):
            parts.append(f"Objective: {example['Objective']}")
        if example.get("Methods"):
            parts.append(f"Methods: {example['Methods']}")
        if example.get("Results"):
            parts.append(f"Results: {example['Results']}")
        return "\n\n".join(parts)

    # Ancien format : clé 'abstract'
    if "abstract" in example:
        return example["abstract"]

    # Fallback : concaténer toutes les valeurs textuelles non-annotation
    skip_keys = {
        "doi", "participatory_method", "participatory_method_rationale",
        "participation_degree", "participant_types",
        "green_infrastructure_intervention", "green_infrastructure_rationale",
        "setting",
    }
    parts = []
    for k, v in example.items():
        if k not in skip_keys and isinstance(v, str) and v:
            parts.append(f"{k}: {v}")
    return "\n\n".join(parts)


def single_shot(example: dict) -> list:
    """
    Create a single few-shot example message pair.

    Args:
        example: Dictionary containing full text sections and expected analysis fields

    Returns:
        List of user and assistant message dictionaries
    """
    full_text = build_fulltext_from_example(example)

    answer = {
        "participatory_method": example["participatory_method"],
        "participatory_method_rationale": example["participatory_method_rationale"],
        "participation_degree": example.get("participation_degree", None),
        "participant_types": example.get("participant_types", None),
        "green_infrastructure_intervention": example["green_infrastructure_intervention"],
        "green_infrastructure_rationale": example["green_infrastructure_rationale"],
        "setting": example.get("setting", None),
    }

    answer_str = json.dumps(answer, ensure_ascii=False)

    return [
        {"role": "user", "content": full_text},
        {"role": "assistant", "content": answer_str},
    ]


def generate_messages(full_text: str, examples: list) -> list:
    """
    Generate few-shot prompt messages for full-text analysis.

    Args:
        full_text: The full text content to analyze
        examples: List of example full texts with expected outputs

    Returns:
        List of message dictionaries for the API call
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for ex in examples:
        messages.extend(single_shot(ex))

    messages.append({"role": "user", "content": full_text})

    return messages

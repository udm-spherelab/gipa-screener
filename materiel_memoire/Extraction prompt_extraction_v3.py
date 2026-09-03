"""
Extraction prompt for the GIPA systematic review — v3 (CSV-driven).
Grid v2.0 — 30 analytic variables across OS1 / OS2 / OS3, plus the A1
study identifier.

The extraction grid is loaded entirely from extraction_grid.csv.
This file only contains fidelity rules, taxonomies, and message helpers.
"""
import json
import csv
import io
from pathlib import Path

# Load grid from CSV
GRID_PATH = Path(__file__).parent / "extraction_grid.csv"

def load_grid(path: Path = GRID_PATH) -> list[dict]:
    """Return the extraction grid as a list of field dicts."""
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

GRID = load_grid()

def grid_key(row: dict) -> str:
    """Return the JSON key to use for a grid row: the Code, or the Variable
    name as a fallback when Code is blank (e.g. Sponsor, Impact of
    participation, Types of justice). Keeps prompt/few-shot/output keys
    consistent with run_extraction.py's KEY_MAP."""
    return row["Code"] or row["Variable"]

# Taxonomies
TAXONOMIES = """
C1 — Intervention type (multiple allowed, separate by ; — if more than one applies,
     list the PRIMARY type first, then secondary type(s) in order of importance):
  1 — New green space creation
      Conversion of previously non-green or hard-surfaced land into a green space. Includes
      parks, pocket parks, community gardens, urban forests on vacant land, daylighted streams,
      and naturalisation of brownfields. Also includes informal-settlement upgrading where
      vegetation is added.
      e.g. new park on former parking lot ; community garden on vacant lot ; urban forest on brownfield ; pocket park on derelict land ; daylighted urban stream
  2 — Modification of existing green space
      Renovation, biodiversity uplift, programming change, or naturalisation of an existing
      park, garden, woodland, cemetery, or other green space. Includes lawn-to-meadow
      conversion in existing parks, native species planting, fence removal, and re-zoning.
      e.g. park renovation/redesign ; lawn-to-native-meadow conversion ; biodiversity planting in existing park ; cemetery naturalisation ; fence removal merging green spaces
  3 — Green-grey hybrid / Nature-Based Infrastructure (NbS)
      Engineered systems combining vegetation with built infrastructure to deliver a
      regulating function (stormwater, heat, pollution). Characterised by a specific
      engineering purpose beyond aesthetics.
      e.g. green roofs and living walls ; bioswales and rain gardens ; sustainable urban drainage (SUDS) ; blue-green infrastructure ; permeable pavement with vegetation
  4 — Streetscape and linear greening
      Greening attached to streets, paths, or other linear corridors. The LINEAR or STREET
      character is the defining feature, not the vegetation type. Includes tactical
      urbanism with vegetation and temporary parklets when made permanent.
      e.g. street trees along sidewalks ; verge/roadside plantings ; parklets ; greenways/green corridors ; hedgerows along roads ; linear cycling path with planting
  5 — Programmatic / stewardship intervention
      Interventions that activate, maintain, or expand ACCESS to greening through PROGRAMMES
      rather than through new physical greening alone. No major new land conversion required.
      Human/organisational in nature.
      e.g. free tree distribution to residents ; park prescriptions by physicians ; 'Friends of the Park' volunteer groups ; community forestry programmes ; urban agriculture support schemes ; environmental education at green sites

  Decision rules (use when a case could fit more than one category):
  1 vs 2 — Was the prior land use non-green? YES → 1. If the prior use was a degraded/existing park → 2.
  2 vs 5 — Physical change to the space itself → 2. Activation/programming with no physical change → 5.
  3 vs 4 — Does it have an engineering/regulating function (stormwater, heat, air quality)? YES → 3.
           Purely ornamental green wall/planting with no regulating function → 4.
  4 vs 1 — Is the LINEAR/STREET character the dominant feature? YES → 4. If trees are part of a new
           park → 1. If a corridor connects two existing parks → 4.

  Coding decision flow (apply in this order, stop at first match):
  1. Is it a programme only, with no land change? → category 5
  2. Is it on a street/path/linear corridor? → category 4
  3. Does it have an engineering/regulating function? → category 3
  4. Was the prior land use non-green? → category 1
  5. Otherwise: existing green space being modified → category 2

C3 — Intervention narrative description:
  Provide a concise free-text description of the intervention as reported in the article.
  State what was created, modified or implemented and, when available, where and for whom.
  The description may contain one or two complete sentences. Do not add information that is
  not supported by the source text.

D4 — Phase(s) of intervention where participation occurred (multiple allowed, separate by ;):
  planning-design — Identification of needs/opportunities, definition of objectives, choice of
      the type of greening arrangement and its location, and initial decision-making on how
      the intervention will be carried out.
  implementation — Execution of the greening works themselves, plus ongoing and long-term
      maintenance of the arrangement once installed.
  monitoring-evaluation — Assessment of the intervention's ecological, health, and social effects,
      including any resident involvement in data collection and in the adaptive revision of the
      arrangement over time.
  Note: participation is most often concentrated in planning-design, less often sustained through
  implementation, and least often integrated into monitoring-evaluation — flag if the article
  explicitly discusses this imbalance (relevant to E3/procedural justice).

E1 — Geekiyanage participatory method taxonomy:
  1 — Consultation, opinion-collection : surveys, interviews, focus groups, public meetings
  2 — Deliberation-expertise          : expert panels, citizen juries, visioning
  3 — Co-design, solution-creation     : workshops, design charrettes, community mapping
  4 — Participatory action, evaluation : ABCD, participatory budgeting, MSC
  → Unknown method: assign best-fit category + note exact name in brackets
  → No fit at all: leave E1 blank and report in E2

E3 — IAP2 level (highest rung substantively reached by the process, not merely offered):
  1-Consult    : the process solicits feedback/suggestions on alternatives, analyses, or decisions,
                 and organisers may adjust their decision in light of that feedback — but the public
                 does not shape the alternatives themselves.
  2-Involve    : organisers work directly with participants throughout the process so that their
                 concerns and aspirations are consistently understood and reflected in the
                 alternatives developed, and participants are told how their input influenced
                 the decision.
  3-Collaborate: participants share responsibility with organisers at every stage of planning and
                 decision-making, including developing alternatives and identifying the preferred
                 solution, with decisions made jointly.
  4-Empower    : decision-making authority and managerial control over the project's development
                 and implementation is delegated to participants/stakeholders.
  (IAP2 Inform excluded per inclusion criteria — one-way information provision with no feedback
  loop does not meet the minimum bar of interactive participation.)

E5 — Type of participants (multiple allowed, separate by ;):
  citizens         — community members directly affected by the intervention
  stakeholders     — individuals/groups with an interest or stake but not necessarily residents
                      (local businesses, advocacy groups, government agencies, NGOs)
  experts          — professionals contributing specialised technical or scientific knowledge
  policy-makers    — those responsible for the final decision (elected officials, project managers)
  students         — school-age or enrolled students engaged specifically as such (e.g. school
                      greening projects, campus interventions) — distinct from general citizens
  supporting-figures — facilitators/staff who run the process without being participants themselves
                      (report separately from participant counts if the article distinguishes them)

E_rec — Selection/recruitment modality (multiple allowed, separate by ;):
  self-selection   — open call; participants opt in on their own initiative
  random-sampling  — recruited via random or random-stratified sampling to broaden representativeness
  targeted-selection — specific individuals/demographic groups invited deliberately (e.g. by age,
                      gender, education, professional role) to increase representativeness
  no-selection     — open/unrestricted access with no recruitment mechanism (e.g. mass media)

E2 — Reference catalogue of participatory techniques (Baldessari et al. 2024, Forests 15:1514,
     Table 1 — 24 techniques identified from >2000 publications). Use this to recognise and name
     techniques consistently, whether they map to an E1 category or not:
  Advertising/Media Coverage — visual/broadcast channels informing audiences (radio, TV, press, web)
  Citizen Committees      — standing group of appointed representatives giving ongoing project advice
  Citizen Juries/Panel    — randomly selected citizens deliberate an issue and reach a consensus
  Idea Collection         — soliciting ideas/suggestions from individuals or groups
  Delphi Method           — iterative anonymous expert questionnaire rounds with feedback until convergence
  Education Events        — workshops/seminars/conferences sharing knowledge with active interaction
  Field Trip              — organised site visit engaging participants with the physical context
  Fishbowl                — small inner-circle group discusses while others observe from outside
  Focus Group             — small selected group discussion exploring attitudes/behaviour on a topic
  Forum                   — structured event where citizens/experts exchange ideas and dialogue
  Inquiry                 — participants co-create research: define questions, gather/analyse/interpret data
  Interview               — structured one-on-one or group conversation gathering insights/opinions
  Most Significant Change — stories of change collected and analysed to identify key themes/impacts
  Poll                    — quick closed-ended question gauging opinion from a sample
  Public Hearing          — formal public presentation; attendees may comment but hold no decision power
  Referendum              — direct public vote on a specific proposal, law, or policy
  Role Game               — participants take on roles in a simulated scenario for learning/problem-solving
  Science Shop            — brokers free/low-cost research expertise access for citizen groups
  Social Media            — platforms used to inform and interactively engage communities
  Survey                  — structured open/closed questionnaire collecting data from a population sample
  Wisdom Council          — small cross-hierarchy group producing recommendations on urgent issues quickly
  Working Group/Expert Panel — stakeholders/experts collaborate to develop strategies or recommendations
  Workshops               — structured hands-on session with discussion/exercises generating ideas/solutions
  World Café              — small-group café-style rotating conversations sharing perspectives on a topic

  Routing rule — avoid double-coding between E1 and E2:
  → If a technique above already appears by name inside an E1 bracketed list, code it under E1
    (with the exact name noted in brackets) — do NOT also list it in E2. This applies to:
    Citizen Committees, Citizen Juries/Panel, Focus Group, Interview, Most Significant Change,
    Poll, Social Media, Survey, Working Group/Expert Panel, Workshops.
  → "Field Trip" = the same concept as "site visits/tours" already listed under E1 category 1 —
    code it there, not in E2.
  → For any other technique in this catalogue (Idea Collection, Delphi Method, Education Events,
    Fishbowl, Forum, Inquiry, Public Hearing, Referendum, Role Game, Science Shop, Wisdom Council,
    World Café): apply E1's own rule first — if how the article actually implements it clearly
    fits one of the 4 E1 categories, code it there with the name in brackets. Only list it in E2
    if it genuinely fits none of the 4 categories.
  → "Advertising/Media Coverage" is INFORM-level only and falls outside the inclusion criteria
    (E3 requires Consult or above) — do not code it as a participatory method at all.

Types of justice (multiple allowed, separate by ;):
  Distributive  : who gets green space benefits / who bears burdens
  Procedural    : fairness of the process (high IAP2 ≠ procedural justice if unrepresentative)
  Recognitional : respect for identities and knowledge systems of marginalised groups
  not-evoked    : justice framing absent from the article
"""

# System prompt
def _build_system_prompt() -> str:
    # Use a CSV writer so commas inside definitions and options remain quoted.
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["Code", "Section", "Variable", "Definition", "Options"])
    for row in GRID:
        writer.writerow([
            grid_key(row), row["Section"], row["Variable"],
            row["Definition"], row["Options"],
        ])
    grid_block = buffer.getvalue().rstrip()

    return f"""You are a systematic-review data extractor specialising in participatory methods and urban greening.

TASK
Extract every field listed in the EXTRACTION GRID below from the article passages provided.
Return a single flat JSON object — one key per variable (use the Code as key) — nothing else.

FIDELITY RULES
- Quote source text directly with page number when possible: "..." (p. X)
- Missing or absent information → "not reported"
- Interpretive judgement beyond the text → flag as [extractor judgement]
- Never invent data from outside the article.

TAXONOMIES
{TAXONOMIES}

EXTRACTION GRID (Code, Section, Variable, Definition, Options)
{grid_block}

Respond with ONLY the JSON object. No markdown, no preamble.
"""

SYSTEM_PROMPT = _build_system_prompt()


def _build_extraction_response_format() -> dict:
    """Return a strict schema requiring one value for every grid field."""
    keys = [grid_key(row) for row in GRID]
    value_schema = {
        "anyOf": [
            {"type": "string"},
            {"type": "array", "items": {"type": "string"}},
        ]
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "gipa_structured_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {key: value_schema for key in keys},
                "required": keys,
                "additionalProperties": False,
            },
        },
    }


EXTRACTION_RESPONSE_FORMAT = _build_extraction_response_format()

# Message builders
def single_shot(example: dict) -> list[dict]:
    """
    Build one few-shot pair from a completed extraction dict.

    Args:
        example: dict with keys 'passages' (str) + one key per grid Code.
    Returns:
        [user_message, assistant_message]
    """
    answer = {grid_key(r): example.get(grid_key(r), "not reported") for r in GRID}
    return [
        {"role": "user",      "content": example.get("passages", "")},
        {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
    ]


def generate_messages(passages: str, examples: list[dict] = None) -> list[dict]:
    """
    Assemble the full message list for one extraction API call.

    Args:
        passages : RAG-retrieved text (chunks joined by '\\n\\n---\\n\\n')
        examples : optional list of completed extraction dicts for few-shot priming
    Returns:
        List of message dicts ready for the OpenAI API.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in (examples or []):
        messages.extend(single_shot(ex))
    messages.append({"role": "user", "content": passages})
    return messages


def build_passages(chunks: list[str]) -> str:
    """Join RAG chunks into a single context string."""
    return "\n\n---\n\n".join(chunks)

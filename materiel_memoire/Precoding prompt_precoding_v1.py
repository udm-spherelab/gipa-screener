"""
Thematic pre-coding prompt for the GIPA systematic review — v1.
Codebook v2 — 34 inductive codes across 5 analytical dimensions.

Unlike the extraction module (prompt_extraction_v3.py), this module does NOT
use RAG: it operates directly on the 5 short passages already isolated by
the extraction phase (3) for each variable — H_succ, H_barr,
impact_participation, conflict (F2d), adverse_effects (F_adv). No chunking,
no embeddings, no retrieval step is required.

The codebook was built inductively by manual line-by-line reading of a
35-article development sample (see MEMOIRE section 2.3, "Thematic
pre-coding"). Coverage saturated at article 16 of that sample for 34 of the
35 codes ultimately observed: the 16 few-shot examples bundled with this
module (few_shot_examples_precoding.json) are exactly those 16 articles,
and jointly cover these 34 codes. The 35th code (ADV0, an explicit
"no adverse effect reported" statement) only emerged later in the wider
corpus and is deliberately NOT seeded in CODEBOOK below — it is left for
the model to (re)discover and propose through the emergent_themes
mechanism, which doubles as a live check that this mechanism actually
works before relying on it for genuinely novel themes.
"""
import json
from pathlib import Path
from typing import Optional

# Codebook
# Five analytical dimensions, one dict per dimension: code -> (label, description)

CODEBOOK = {
    "H_succ": {
        "SUC1a": ("Institutional/political support",
                  "Formal partnerships, political leadership, policy integration, "
                  "municipal/governmental support (excluding funding)."),
        "SUC1b": ("Financial resources",
                  "Grants, stable funding, sponsorship, donations."),
        "SUC1c": ("Secure land tenure",
                  "Secured tenure, formal land agreement, site access."),
        "SUC2":  ("Social capital & trust",
                  "Networks, cooperation, cohesion, trust among actors."),
        "SUC3":  ("Participatory process design",
                  "Co-design, flexibility, low entry barriers, adaptive structure."),
        "SUC4":  ("Leadership / continuity",
                  "Individual or collective leadership, facilitation, succession."),
        "SUC5":  ("Communication / education / outreach",
                  "Educational programmes, clear communication, awareness-raising."),
        "SUC6":  ("Recognition of participant agency",
                  "Choice, autonomy, strengths-based approach."),
        "SUC7":  ("Local fit / contextual relevance",
                  "Proximity, visibility, fit with local needs and site."),
    },
    "H_barr": {
        "BAR0":  ("No specific barrier reported",
                  "Text explicitly states the absence of a participation-related barrier."),
        "BAR1":  ("Institutional / top-down governance constraints",
                  "Bureaucracy, unclear mandates, tokenistic consultation, top-down process."),
        "BAR2a": ("Financial constraints",
                  "Unstable/insufficient funding, financial dependency."),
        "BAR2b": ("Labour / time constraints",
                  "Unstable volunteering, recruitment, availability, tight deadlines."),
        "BAR3":  ("Land tenure / legal insecurity",
                  "Uncertain tenure, no formal ownership, threat of reclamation "
                  "(mirrors SUC1c)."),
        "BAR4":  ("Representativeness / inclusion deficits",
                  "Self-selection, exclusion of groups, language, physical access."),
        "BAR5":  ("Knowledge / skills deficits",
                  "Lack of expertise, low public understanding."),
        "BAR6":  ("Distrust / power asymmetries",
                  "Scepticism toward participatory \"rhetoric\", unequal power."),
        "BAR7":  ("External shocks / context",
                  "COVID-19, seasonality, weather conditions."),
        "BAR8":  ("Continuity / sustainability fragility",
                  "Burnout, dependency on leaders, ageing membership."),
    },
    "impact_participation": {
        "IMP1": ("Social capital, cohesion, collective learning",
                 "Trust, social ties, mutual learning."),
        "IMP2": ("Empowerment / ownership",
                 "Autonomy, sense of belonging, resistance to stereotypes."),
        "IMP3": ("Influence on decision / design",
                 "Shaped the outcome, priorities, final plan."),
        "IMP4": ("Institutional / behavioural change",
                 "Change among officials, professionals, designers."),
        "IMP5": ("Limited, unmeasured, or undemonstrated impact",
                 "Effect not evaluated, restricted scope, purely consultative role."),
        "IMP6": ("Skills / education gain",
                 "Transferable skills, learning outcomes."),
    },
    "conflict": {
        "CON0": ("Explicitly no conflict reported", ""),
        "CON1": ("Community vs. authority/institution",
                 "Decision-making power, land use, municipal management."),
        "CON2": ("Intra-community conflict",
                 "Internal norms, internal exclusion, gatekeeping."),
        "CON3": ("Livelihood/informal-use tension vs. conservation-formalisation", ""),
        "CON4": ("Disputes among stakeholders over priorities",
                 "Resource allocation, diverging prioritisation."),
    },
    "adverse_effects": {
        "ADV1": ("Displacement / exclusion / gentrification", ""),
        "ADV2": ("Environmental / sanitary / safety nuisances",
                 "Waste, pests, hazards, noise."),
        "ADV3": ("Governance-related harms",
                 "Loss of autonomy, power asymmetry, instrumentalisation."),
        "ADV4": ("Social exclusion dynamics",
                 "Soft exclusion, parochialism, exclusionary fencing."),
    },
}

NR_CODE = "NR"  # applies to all five dimensions: "not reported" in the source passage

DIMENSIONS = list(CODEBOOK.keys())  # ["H_succ", "H_barr", "impact_participation", "conflict", "adverse_effects"]

# Codebook rendering for the system prompt

def _render_codebook() -> str:
    blocks = []
    for dim, codes in CODEBOOK.items():
        lines = [f"\n{dim} — allowed codes:"]
        for code, (label, desc) in codes.items():
            lines.append(f"  {code} — {label}{': ' + desc if desc else ''}")
        lines.append(f"  {NR_CODE} — not reported / no information available in the passage")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)

CODEBOOK_TEXT = _render_codebook()

# System prompt

def _build_system_prompt() -> str:
    return f"""You are a qualitative-coding assistant specialising in participatory methods and urban greening.

TASK
You will receive the five short passages already isolated for one article during the extraction
phase of a systematic review: H_succ (success factors), H_barr (barriers), impact_participation,
conflict, and adverse_effects. For EACH of these five passages, assign every code from the
CODEBOOK below that applies. Multiple codes per passage are expected and normal.

CODING RULES
- Base every code strictly on the passage text provided — never invent content from outside it.
- If a passage's text is exactly "not reported" (or equivalent, no information available),
  return only "{NR_CODE}" for that dimension.
- If a passage explicitly states the ABSENCE of something (e.g. the barrier or conflict text
  says there is none), use that dimension's explicit-absence code (BAR0 / CON0) rather than
  "{NR_CODE}". "{NR_CODE}" is reserved for missing/unreported data, not for a stated absence.
  The adverse_effects dimension currently has no such explicit-absence code: if a passage
  clearly states that no adverse effect was observed, do not force-fit "{NR_CODE}" — treat it
  as a candidate emergent theme instead (see below).
- Return codes as a comma-separated string, e.g. "SUC1a, SUC2, SUC7".
- A codebook built inductively from a development sample cannot guarantee exhaustive
  theoretical saturation over the full corpus. If a passage clearly describes a recurring,
  substantive pattern that does NOT fit any existing code in its dimension, do not force-fit
  it: instead, add a short entry to "emergent_themes" — {{"dimension": ..., "candidate_label":
  ..., "quote": ...}} — and still assign the closest existing code(s) if any partially apply
  (or "{NR_CODE}" if none do). Do not use "emergent_themes" for content that already fits an
  existing code reasonably well, even imperfectly — reserve it for genuinely new, recurring
  patterns.

CODEBOOK
{CODEBOOK_TEXT}

OUTPUT FORMAT
Return a single flat JSON object with exactly these keys: "H_succ", "H_barr",
"impact_participation", "conflict", "adverse_effects" (each a comma-separated code string),
and "emergent_themes" (a list, empty if none). No markdown, no preamble, JSON only.
"""

SYSTEM_PROMPT = _build_system_prompt()

# Few-shot loading

EXAMPLES_PATH = Path(__file__).parent / "few_shot_examples_precoding.json"

def load_examples(path: Path = EXAMPLES_PATH) -> list[dict]:
    """Load the bundled few-shot examples (16 articles covering all 35 codes)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# Message builders

def single_shot(example: dict) -> list[dict]:
    """
    Build one few-shot pair from a manually coded article.

    Args:
        example: dict with keys 'passages' (str) + one key per dimension in
                  DIMENSIONS + optionally 'emergent_themes' (list).
    Returns:
        [user_message, assistant_message]
    """
    answer = {dim: example.get(dim, NR_CODE) for dim in DIMENSIONS}
    answer["emergent_themes"] = example.get("emergent_themes", [])
    return [
        {"role": "user",      "content": example["passages"]},
        {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
    ]


def build_input(row: dict) -> str:
    """
    Build the 'passages' input string from one row of the extraction CSV
    (a dict with the five OS3 text columns, already extracted in phase 3).

    Args:
        row: dict with keys "H_succ Success factors identified",
             "H_barr Barriers identified", "Impact of participation",
             "F2d Conflict and contestation",
             "F_adv Reported unintended adverse effects"
             (i.e. the raw TEMPLATE_COLUMNS names from the extraction module).
    Returns:
        A single labeled text block, in the same format as the bundled
        few-shot examples.
    """
    return (
        f"H_succ: {row.get('H_succ Success factors identified', 'not reported')}\n\n"
        f"H_barr: {row.get('H_barr Barriers identified', 'not reported')}\n\n"
        f"impact_participation: {row.get('Impact of participation', 'not reported')}\n\n"
        f"conflict: {row.get('F2d Conflict and contestation', 'not reported')}\n\n"
        f"adverse_effects: {row.get('F_adv Reported unintended adverse effects', 'not reported')}"
    )


def generate_messages(passages: str, examples: Optional[list[dict]] = None) -> list[dict]:
    """
    Assemble the full message list for one thematic pre-coding API call.

    Args:
        passages : the labeled text block built by build_input()
        examples : optional list of manually coded articles for few-shot priming
                   (defaults to the bundled 16-article set covering all 35 codes)
    Returns:
        List of message dicts ready for the OpenAI API.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in (examples if examples is not None else load_examples()):
        messages.extend(single_shot(ex))
    messages.append({"role": "user", "content": passages})
    return messages

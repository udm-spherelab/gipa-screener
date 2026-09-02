"""
Pipeline de screening full-text (méthodes participatives / verdissement urbain)
— script unique, sans dépendance à GROBID.

Le texte de chaque article est extrait directement du PDF via PyMuPDF (fitz),
comme dans run_extraction.py. Le prompt (SYSTEM_PROMPT + génération des
messages few-shot) reste dans un fichier séparé : src/prompt.py.

Structure attendue à côté de ce script :
    full_texts_pdf/        les PDF à screener
    in/exemples.json       les exemples few-shot annotés
    out/                   résultats + log + checkpoint (créé si absent)
    src/prompt.py          SYSTEM_PROMPT + generate_messages()
"""

import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
import polars as pl
from openai import OpenAI
from pydantic import BaseModel

from src.prompt import SYSTEM_PROMPT, generate_messages

# ── Chemins du projet ────────────────────────────────────────────────────────

PROJECT_ROOT    = Path(__file__).parent.absolute()
FULL_TEXTS_DIR  = PROJECT_ROOT / "full_texts_pdf"
INPUT_DIR       = PROJECT_ROOT / "in"
OUTPUT_DIR      = PROJECT_ROOT / "out"

EXAMPLES_FILE   = INPUT_DIR / "exemples.json"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint_screening.json"

OUTPUT_DIR.mkdir(exist_ok=True)

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_NAME          = "gpt-5-nano-2025-08-07"
CHECKPOINT_INTERVAL = 1

# Garde-fou de taille de contexte. Le bloc fixe (system prompt + exemples
# few-shot) est envoyé à CHAQUE appel et peut à lui seul représenter une part
# importante de la fenêtre de contexte du modèle (ex. ~190k tokens pour 40
# exemples volumineux). On calcule dynamiquement la marge restante pour le
# texte de l'article et on tronque si nécessaire, plutôt que de laisser
# l'appel échouer avec une erreur 400 sur les articles les plus longs.
MAX_CHARS_CONTEXT = 1_400_000  # ~350k tokens de marge (fenêtre gpt-5-nano : 400k)

logger = logging.getLogger(__name__)


# ── Schéma de sortie structuré ───────────────────────────────────────────────

class Output(BaseModel):
    """Schéma de sortie pour l'analyse full-text."""

    participatory_method: bool | None
    participatory_method_rationale: str
    participation_degree: str | None
    participant_types: str | None
    green_infrastructure_intervention: bool | None
    green_infrastructure_rationale: str
    setting: str | None


# ── Nettoyage du texte extrait du PDF ────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Supprime les sections non pertinentes pour le screening :
    - Références / Bibliographie / Annexes (comme dans run_extraction.py)
    - Le bloc administratif de fin d'article (Author Contributions, Funding,
      Acknowledgments, Conflicts of Interest, Data Availability Statement,
      Supplementary Materials...) qui précède généralement les Références
      dans les articles MDPI — sans intérêt pour la décision de screening,
      et qui gonfle inutilement le nombre de tokens envoyés au modèle.
    Conserve les lignes contenant un DOI (jamais supprimées, même isolées).
    """
    text = re.split(
        r'\n\s*(References|Bibliography|REFERENCES|Références|Works Cited)\s*\n',
        text
    )[0]
    text = re.split(
        r'\n\s*(Appendix|Annexe|APPENDIX)\s*[A-Z]?\s*\n',
        text
    )[0]
    text = re.split(
        r'\n\s*(Author Contributions|Funding|Institutional Review Board Statement|'
        r'Informed Consent Statement|Data Availability Statement|Acknowledgments|'
        r'Acknowledgements|Conflicts of Interest|Conflict of Interest|'
        r'Declaration of Competing Interest|Supplementary Materials)\s*:?\s*\n',
        text
    )[0]
    lines = [
        l for l in text.split('\n')
        if not (re.match(r'^\s*https?://\S+\s*$', l) and 'doi.org' not in l.lower())
    ]
    return '\n'.join(lines)


# ── Extraction robuste d'un DOI (identique à run_extraction.py) ─────────────

def _extract_doi_from_text(text: str):
    """Voir run_extraction.py pour le détail du raisonnement derrière cette
    fonction (gestion des coupures de ligne dues à la justification PDF)."""
    m = re.search(r"doi\.org/|doi:\s*10\.", text, re.IGNORECASE)
    if not m:
        return None
    window = text[m.start():m.start() + 200]
    m2 = re.search(r"(?:doi\.org/|doi:\s*)(10\.\d{4,9}/\S*)", window, re.IGNORECASE)
    if not m2:
        return None
    suffix = m2.group(1)
    if suffix == "" or suffix.endswith("/"):
        rest = window[m2.end():].lstrip("\r\n ")
        next_line = rest.split("\n", 1)[0]
        continuation = next_line.split()[0] if next_line.split() else ""
        suffix += continuation
    if not suffix or suffix.endswith("/"):
        return None
    return "https://doi.org/" + suffix.rstrip(".,);]")


def extract_doi_from_pdf(doc) -> str | None:
    """
    DOI propre à l'article : texte des 2 premières pages en priorité (là où
    les éditeurs impriment leur propre DOI), lien hypertexte des 2 premières
    pages en secours — jamais toute la bibliographie, pour ne pas capter le
    DOI d'une référence citée via un lien [CrossRef].
    """
    first_pages_text = "".join(p.get_text() for p in doc[:2])
    doi = _extract_doi_from_text(first_pages_text)
    if doi:
        return doi
    try:
        for page in doc[:2]:
            for link in page.get_links():
                uri = link.get("uri", "") or ""
                if "doi.org" in uri.lower():
                    return uri.strip().rstrip(".,);]")
    except Exception:
        pass
    return None


# ── Chargement des exemples et du checkpoint ────────────────────────────────

def load_examples(examples_file: Path) -> list:
    if not examples_file.exists():
        logger.error(f"Fichier d'exemples introuvable : {examples_file}")
        sys.exit(1)
    try:
        with open(examples_file, "r", encoding="utf-8") as f:
            examples = json.loads(f.read(), strict=False)
    except json.JSONDecodeError as e:
        logger.error(f"JSON invalide dans le fichier d'exemples : {e}")
        sys.exit(1)

    if not isinstance(examples, list):
        logger.error("Le fichier d'exemples doit contenir un tableau JSON")
        sys.exit(1)

    required_fields = [
        "participatory_method", "participatory_method_rationale",
        "participation_degree", "participant_types",
        "green_infrastructure_intervention", "green_infrastructure_rationale",
        "setting",
    ]
    for i, example in enumerate(examples):
        for field in required_fields:
            if field not in example:
                logger.error(f"Exemple {i} : champ requis manquant '{field}'")
                sys.exit(1)

    logger.info(f"{len(examples)} exemples chargés depuis {examples_file}")
    return examples


def load_checkpoint(checkpoint_file: Path) -> list:
    if checkpoint_file.exists():
        logger.info(f"Reprise depuis le checkpoint : {checkpoint_file}")
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        logger.info(f"{len(results)} articles déjà traités")
        return results
    return []


def save_checkpoint(checkpoint_file: Path, results: list) -> None:
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ── Analyse d'un article ─────────────────────────────────────────────────────

def analyze_fulltext(full_text: str, examples: list, model: str, client: OpenAI) -> dict:
    messages = generate_messages(full_text, examples)
    response = client.responses.parse(
        model=model,
        input=messages,
        text_format=Output,
        reasoning={"effort": "high"},
    )
    return response.output_parsed.model_dump()


def compute_rating(result: dict) -> int:
    """
    Reproduit exactement la définition du Tableau 6 du mémoire :
    - 3 : les deux critères sont vrais ET contexte urbain OU péri-urbain
    - 2 : au moins un critère est None/incertain, OU les deux critères sont
          vrais mais le contexte est "mixed" ou indéterminé (relecture
          manuelle nécessaire)
    - 1 : au moins un critère est faux, OU les deux sont vrais mais le
          contexte est rural
    """
    pm = result["participatory_method"]
    gi = result["green_infrastructure_intervention"]
    setting = result.get("setting")

    if pm is None or gi is None:
        return 2
    if not (pm and gi):
        return 1
    if setting in ("urban", "peri_urban"):
        return 3
    if setting == "rural":
        return 1
    # setting == "mixed", ou None/indéterminé alors que pm et gi sont
    # tranchés : classification incertaine -> relecture manuelle
    return 2


# ── Pipeline principal ───────────────────────────────────────────────────────

def run_pipeline():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(OUTPUT_DIR / "screening_fulltext.log"),
            logging.StreamHandler(),
        ],
    )

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY n'est pas définie")
        logger.error("Définissez-la avec : export OPENAI_API_KEY='votre-clé'")
        sys.exit(1)

    client = OpenAI()
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"fulltext_screened_by_{MODEL_NAME}-{now}.csv"

    examples = load_examples(EXAMPLES_FILE)

    # Taille du bloc fixe (system prompt + exemples), réutilisé identique à
    # chaque appel — sert à dimensionner la marge disponible pour le texte
    # de chaque article et à décider s'il faut le tronquer.
    fixed_chars = len(SYSTEM_PROMPT) + sum(
        len(json.dumps(ex, ensure_ascii=False)) for ex in examples
    )
    logger.info(f"Bloc fixe (prompt + exemples) : ~{fixed_chars // 4} tokens estimés")

    results       = load_checkpoint(CHECKPOINT_FILE)
    processed_ids = {r["doi"] for r in results}

    files = sorted(
        list(FULL_TEXTS_DIR.glob("*.pdf")) + list(FULL_TEXTS_DIR.glob("*.txt"))
    )
    logger.info(f"{len(files)} fichiers trouvés dans {FULL_TEXTS_DIR}")

    counter = len(results)
    failed_files = []

    for f in files:
        article_id = f.stem

        # Protéger l'ouverture/lecture de CHAQUE fichier individuellement.
        # Un PDF corrompu, tronqué, ou mal téléchargé (ex. page d'erreur
        # d'accès enregistrée à la place de l'article) ne doit jamais faire
        # planter tout le run — surtout après plusieurs dizaines de minutes
        # de traitement déjà effectué. On log l'échec et on passe au suivant.
        try:
            if f.suffix == ".pdf":
                doc  = fitz.open(str(f))
                text = "\n".join(p.get_text() for p in doc)
                doi  = extract_doi_from_pdf(doc) or article_id
            else:
                text = f.read_text(encoding="utf-8", errors="ignore")
                doi  = article_id
        except Exception as e:
            logger.error(f"  Impossible d'ouvrir/lire {f.name} : {e}")
            failed_files.append({"file": f.name, "error": str(e)})
            continue

        if doi in processed_ids:
            continue

        counter += 1
        logger.info(f"Analyse #{counter} : {article_id} (DOI : {doi})")

        full_text = clean_text(text)
        if len(full_text.split()) < 200:
            logger.warning(f"  Texte trop court après nettoyage, ignoré : {article_id}")
            continue

        # Troncature de sécurité si le texte de l'article, ajouté au bloc
        # fixe (prompt + exemples), dépasserait la marge de contexte. On
        # garde le début (résumé/intro) et la fin (résultats/conclusion) de
        # l'article, qui concentrent généralement l'information utile au
        # screening — plutôt que de laisser l'appel échouer.
        available_chars = MAX_CHARS_CONTEXT - fixed_chars
        if len(full_text) > available_chars > 0:
            half = available_chars // 2
            full_text = (
                full_text[:half]
                + "\n\n[...TEXTE TRONQUÉ POUR RESPECTER LA LIMITE DE CONTEXTE...]\n\n"
                + full_text[-half:]
            )
            logger.warning(f"  Texte tronqué pour {article_id} (article très long)")

        try:
            result = analyze_fulltext(full_text, examples, model=MODEL_NAME, client=client)
            result_dict = {
                "doi": doi,
                "article_id": article_id,
                **result,
                "rating": compute_rating(result),
            }
            results.append(result_dict)
        except Exception as e:
            logger.error(f"  Échec de l'analyse pour {article_id} : {e}")
            save_checkpoint(CHECKPOINT_FILE, results)
            logger.info(f"  Checkpoint sauvegardé après erreur à l'article #{counter}")
            continue

        if counter % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(CHECKPOINT_FILE, results)
            logger.info(f"  Checkpoint sauvegardé à l'article #{counter}")

    pl.DataFrame(results).write_csv(output_file)
    logger.info(f"Screening terminé. Résultats : {output_file}")

    if failed_files:
        failed_report = OUTPUT_DIR / f"fichiers_illisibles_{now}.json"
        with open(failed_report, "w", encoding="utf-8") as f:
            json.dump(failed_files, f, indent=2, ensure_ascii=False)
        logger.warning(
            f"{len(failed_files)} fichier(s) n'ont pas pu être ouverts/lus — "
            f"détail dans {failed_report}. Ces PDF sont probablement corrompus "
            f"ou mal téléchargés (page d'erreur d'accès enregistrée à la place "
            f"de l'article) ; il faut les re-télécharger."
        )

    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("Fichier de checkpoint supprimé après succès")


if __name__ == "__main__":
    run_pipeline()
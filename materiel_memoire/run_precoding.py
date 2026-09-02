"""
Runner — GIPA-IA phase 4: thematic pre-coding.

Reads the extraction CSV already produced by run_extraction.py (phase 3),
and for each article, classifies its 5 OS3 text passages (H_succ, H_barr,
impact_participation, conflict, adverse_effects) against the inductive
codebook defined in prompt_precoding_v1.py.

No RAG, no chunking, no embeddings, no PDF parsing: the input is the
already-extracted short passages, not the full-text article. See
prompt_precoding_v1.py's module docstring and MEMOIRE section 2.3
("Thematic pre-coding") for the rationale.

Usage:
    python3 run_precoding.py
    (edit the paths in the __main__ block, or import run_pipeline()
    and call it directly with your own paths)
"""
import os, csv, json
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
import sys
sys.path.insert(0, str(Path(__file__).parent))
from prompt_precoding_v1 import generate_messages, build_input, load_examples, DIMENSIONS

load_dotenv()


class QuotaExhaustedError(Exception):
    """Levée quand l'API retourne une erreur de quota / crédits insuffisants."""
    pass


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL  = os.getenv("PRECODING_MODEL", "gpt-5.5")

# Colonnes lues depuis le CSV d'extraction (phase 3)
# Noms exacts des colonnes OS3 pertinentes dans le template d'extraction.

INPUT_COLUMNS = {
    "H_succ Success factors identified",
    "H_barr Barriers identified",
    "Impact of participation",
    "F2d Conflict and contestation",
    "F_adv Reported unintended adverse effects",
}

# Colonnes du CSV de sortie (ordre exact)

OUTPUT_COLUMNS = [
    "H_succ_codes",
    "H_barr_codes",
    "impact_participation_codes",
    "conflict_codes",
    "adverse_effects_codes",
    "emergent_themes",
]

# Initialiser le CSV de sortie avec les en-têtes (si fichier n'existe pas)

def _init_csv_if_needed(output_path: str):
    path = Path(output_path)
    if path.exists() and path.stat().st_size > 0:
        return  # déjà initialisé
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["_article_id", "author_year"] + OUTPUT_COLUMNS)

# Charger les article_id déjà pré-codés (reprise sur interruption)

def _load_done_ids(output_path: str) -> set[str]:
    done = set()
    path = Path(output_path)
    if not path.exists():
        return done
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("_article_id", "").strip():
                done.add(row["_article_id"].strip())
    return done

# Ajouter une ligne au CSV de sortie (écriture incrémentale)

def _append_row(output_path: str, article_id: str, author_year: str, result: dict):
    with open(output_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        emergent = result.get("emergent_themes", [])
        writer.writerow(
            [article_id, author_year]
            + [result.get(dim, "NR") for dim in DIMENSIONS]
            + [json.dumps(emergent, ensure_ascii=False) if emergent else ""]
        )

# Journal séparé des thèmes émergents proposés (pour revue par R1)

def _log_emergent_themes(log_path: str, article_id: str, author_year: str, emergent: list):
    if not emergent:
        return
    path = Path(log_path)
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["_article_id", "author_year", "dimension", "candidate_label", "quote"])
        for item in emergent:
            writer.writerow([
                article_id, author_year,
                item.get("dimension", ""),
                item.get("candidate_label", ""),
                item.get("quote", ""),
            ])
    print(f"{len(emergent)} thème(s) émergent(s) proposé(s) — journalisé(s) dans {log_path}")

# Lire le CSV d'extraction (3 lignes d'en-tête)

def _read_extraction_csv(extraction_csv_path: str) -> list[dict]:
    """
    Lit le CSV d'extraction produit par run_extraction.py (3 lignes d'en-tête :
    titre, section, puis noms de colonnes) et retourne une liste de dicts,
    une par article, avec _article_id inclus.
    """
    with open(extraction_csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)  # ligne 1 : titre du template
        next(reader)  # ligne 2 : sections (OS1/OS2/OS3)
        col_names = next(reader)  # ligne 3 : noms de colonnes exacts
        rows = [dict(zip(col_names, r)) for r in reader if any(r)]
    return rows

# Pré-coder un article

def precode_article(row: dict, examples: Optional[list] = None) -> dict:
    passages = build_input(row)
    messages = generate_messages(passages, examples)

    try:
        response = client.chat.completions.create(
            model            = MODEL,
            messages         = messages,  # type: ignore[arg-type]
            reasoning_effort = "medium",
            response_format  = {"type": "json_object"},
        )
    except Exception as e:
        err_msg = str(e).lower()
        if "insufficient_quota" in err_msg or "billing" in err_msg or \
           "exceeded" in err_msg or "quota" in err_msg:
            raise QuotaExhaustedError(f"Crédits API épuisés : {e}")
        raise

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("Réponse vide du modèle (content=None) — impossible de parser le JSON.")
    result = json.loads(content)

    # Garde-fou : s'assurer que les 5 dimensions sont présentes, sinon NR
    for dim in DIMENSIONS:
        if not str(result.get(dim, "")).strip():
            result[dim] = "NR"

    return result

# Pipeline complet

def run_pipeline(extraction_csv_path: str, output_path: str, emergent_log_path: Optional[str] = None,
                  examples_path: Optional[str] = None):
    """
    Args:
        extraction_csv_path : chemin du CSV produit par run_extraction.py (phase 3)
        output_path         : chemin du CSV de sortie du pré-codage thématique (phase 4)
        emergent_log_path   : chemin du journal des thèmes émergents proposés
                               (défaut : à côté de output_path)
        examples_path       : chemin des exemples few-shot (défaut : les 16 exemples
                               fournis, couvrant les 35 codes — voir
                               few_shot_examples_precoding.json)
    """
    if emergent_log_path is None:
        emergent_log_path = str(Path(output_path).with_name("emergent_themes_log.csv"))

    examples = load_examples(Path(examples_path)) if examples_path else load_examples()
    print(f"{len(examples)} exemples few-shot chargés")

    _init_csv_if_needed(output_path)
    done_ids = _load_done_ids(output_path)
    if done_ids:
        print(f"{len(done_ids)} articles déjà pré-codés — ils seront ignorés")

    rows = _read_extraction_csv(extraction_csv_path)
    print(f"{len(rows)} articles trouvés dans {extraction_csv_path}")

    n_coded, n_skipped, n_incomplete = 0, 0, 0

    for row in rows:
        article_id  = row.get("_article_id", "").strip()
        author_year = row.get("A2+A3 Author(s) and year", "").strip()

        if not article_id:
            continue
        if article_id in done_ids:
            n_skipped += 1
            continue

        # Ignorer uniquement les articles dont l'extraction n'a manifestement
        # pas encore eu lieu (les 5 champs OS3 tous vides/"not reported" à la
        # fois). Un seul champ à "not reported" est une valeur normale
        # (ex. F_adv est souvent légitimement non rapporté) et ne doit
        # surtout pas faire écarter l'article — c'est justement au modèle
        # de coder ce champ-là "NR".
        if all(row.get(c, "not reported").strip().lower() in ("", "not reported")
               for c in INPUT_COLUMNS):
            n_incomplete += 1
            continue

        print(f"\n{article_id} ({author_year})")

        try:
            result = precode_article(row, examples)
            _append_row(output_path, article_id, author_year, result)
            _log_emergent_themes(emergent_log_path, article_id, author_year,
                                  result.get("emergent_themes", []))
            n_coded += 1
            print(f"Pré-codé et sauvegardé ({n_coded} total)")

        except QuotaExhaustedError as e:
            print(f"\nARRÊT — {e}")
            print(f"{n_coded} articles pré-codés cette session.")
            print(f"Recharge tes crédits puis relance le script — il reprendra automatiquement.")
            return

        except Exception as e:
            print(f"Erreur : {e}")

    print(f"\nPré-codage thématique terminé : {n_coded} nouveaux + {n_skipped} déjà faits "
          f"+ {n_incomplete} en attente d'extraction")

# Point d'entrée

if __name__ == "__main__":
    run_pipeline(
        extraction_csv_path = "./resultats_extraction.csv",
        output_path         = "./resultats_precoding.csv",
    )
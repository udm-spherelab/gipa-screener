import os, json, csv, re
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
import sys
sys.path.insert(0, str(Path(__file__).parent))
from prompt_extraction import generate_messages, build_passages, GRID

load_dotenv()

# Exception personnalisée pour arrêt propre sur quota épuisé
class QuotaExhaustedError(Exception):
    """Levée quand l'API retourne une erreur de quota / crédits insuffisants."""
    pass

client     = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma     = chromadb.PersistentClient(path=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
collection = chroma.get_or_create_collection("gipa_articles", metadata={"hnsw:space": "cosine"})
MODEL      = os.getenv("EXTRACTION_MODEL", "gpt-5.5")

# Colonnes du template (ordre exact)
TEMPLATE_COLUMNS = [
    "A1 Study ID: DOI / URL",
    "A2+A3 Author(s) and year",
    "A4 Journal / source",
    "A6+A7 Country and city/locality",
    "A8 Urban context type",
    "B1 Study design",
    "B4 Funding",
    "Sponsor",
    "C1 Type of greening intervention",
    "C1 rationale",
    "C3 Intervention narrative description",
    "C5 Stated goals / co-benefits",
    "D1 Spatial scale of intervention",
    "D4 Phase(s) of participation",
    "E1 Participatory method type [+ Geekiyanage taxonomy]",
    "E1 rationale",
    "E2 Other methods reported",
    "E3 IAP2 level (depth of participation)",
    "E3 rationale",
    "E5 Type and profile of participants",
    "E_rec Recruitment modalities",
    "E_gov Governance of participation",
    "E_gov rationale",
    "E_eng Temporal engagement (duration + frequency)",
    "E_n Number of participants",
    "E_rate Participation rate",
    "Impact of participation",
    "Impact of intervention",
    "H_succ Success factors identified",
    "H_barr Barriers identified",
    "Types of justice",
    "Types of justice rationale",
    "F1d Beneficiary groups",
    "F2a Representativeness and diversity of process",
    "F2a rationale",
    "F2d Conflict and contestation",
    "F3c Identity and representation",
    "F_adv Reported unintended adverse effects",
]

SECTION_MAP = {
    "A1 Study ID: DOI / URL":                                "OS1",
    "A2+A3 Author(s) and year":                             "OS1",
    "A4 Journal / source":                                  "OS1",
    "A6+A7 Country and city/locality":                      "OS1",
    "A8 Urban context type":                                "OS1",
    "B1 Study design":                                      "OS1",
    "B4 Funding":                                           "OS1",
    "Sponsor":                                              "OS1",
    "C1 Type of greening intervention":                     "OS1",
    "C1 rationale":                                         "OS1",
    "C3 Intervention narrative description":                "OS1",
    "C5 Stated goals / co-benefits":                        "OS1",
    "D1 Spatial scale of intervention":                     "OS1",
    "D4 Phase(s) of participation":                         "OS2",
    "E1 Participatory method type [+ Geekiyanage taxonomy]":"OS2",
    "E1 rationale":                                         "OS2",
    "E2 Other methods reported":                            "OS2",
    "E3 IAP2 level (depth of participation)":               "OS2",
    "E3 rationale":                                         "OS2",
    "E5 Type and profile of participants":                  "OS2",
    "E_rec Recruitment modalities":                         "OS2",
    "E_gov Governance of participation":                    "OS2",
    "E_gov rationale":                                      "OS2",
    "E_eng Temporal engagement (duration + frequency)":     "OS2",
    "E_n Number of participants":                           "OS2",
    "E_rate Participation rate":                            "OS2",
    "Impact of participation":                              "OS3",
    "Impact of intervention":                               "OS3",
    "H_succ Success factors identified":                    "OS3",
    "H_barr Barriers identified":                           "OS3",
    "Types of justice":                                     "OS3",
    "Types of justice rationale":                            "OS3",
    "F1d Beneficiary groups":                               "OS3",
    "F2a Representativeness and diversity of process":      "OS3",
    "F2a rationale":                                        "OS3",
    "F2d Conflict and contestation":                        "OS3",
    "F3c Identity and representation":                      "OS3",
    "F_adv Reported unintended adverse effects":            "OS3",
}

KEY_MAP = {
    "A1":                      "A1 Study ID: DOI / URL",
    "A2+A3":                   "A2+A3 Author(s) and year",
    "A4":                      "A4 Journal / source",
    "A6+A7":                   "A6+A7 Country and city/locality",
    "A8":                      "A8 Urban context type",
    "B1":                      "B1 Study design",
    "B4":                      "B4 Funding",
    "Sponsor":                 "Sponsor",
    "C1":                      "C1 Type of greening intervention",
    "C1_r":                    "C1 rationale",
    "C3":                      "C3 Intervention narrative description",
    "C5":                      "C5 Stated goals / co-benefits",
    "D1":                      "D1 Spatial scale of intervention",
    "D4":                      "D4 Phase(s) of participation",
    "E1":                      "E1 Participatory method type [+ Geekiyanage taxonomy]",
    "E1_r":                    "E1 rationale",
    "E2":                      "E2 Other methods reported",
    "E3":                      "E3 IAP2 level (depth of participation)",
    "E3_r":                    "E3 rationale",
    "E5":                      "E5 Type and profile of participants",
    "E_rec":                   "E_rec Recruitment modalities",
    "E_gov":                   "E_gov Governance of participation",
    "E_gov_r":                 "E_gov rationale",
    "E_eng":                   "E_eng Temporal engagement (duration + frequency)",
    "E_n":                     "E_n Number of participants",
    "E_rate":                  "E_rate Participation rate",
    "Impact of participation": "Impact of participation",
    "H1":                      "Impact of intervention",
    "H_succ":                  "H_succ Success factors identified",
    "H_barr":                  "H_barr Barriers identified",
    "Types of justice":        "Types of justice",
    "Tj_r":                    "Types of justice rationale",
    "F1d":                     "F1d Beneficiary groups",
    "F2a":                     "F2a Representativeness and diversity of process",
    "F2a_r":                   "F2a rationale",
    "F2d":                     "F2d Conflict and contestation",
    "F3c":                     "F3c Identity and representation",
    "F_adv":                   "F_adv Reported unintended adverse effects",
}

# Champs critiques qui déclenchent un fallback si "not reported"
CRITICAL_FIELDS = {
    "A1":    "What is the DOI or URL of this article? "
             "Look for 'https://doi.org/' or 'doi.org/' in the citation block, header, or footer.",
    "A2+A3": "Who are the authors of this article and what year was it published? "
             "Look for names near the title, in the byline, or in the citation block.",
    "A4":    "What is the name of the journal or publication venue of this article?",
    "A6+A7": "In which country and city was this study conducted?",
}

# Nettoyage du texte avant indexation
def clean_text(text: str) -> str:
    """Supprimer références, annexes et lignes URL isolées."""
    text = re.split(
        r'\n\s*(References|Bibliography|REFERENCES|Références|Works Cited)\s*\n',
        text
    )[0]
    text = re.split(
        r'\n\s*(Appendix|Annexe|APPENDIX)\s*[A-Z]?\s*\n',
        text
    )[0]
    # NB: on ne supprime plus les lignes contenant un DOI, même isolées sur
    # leur propre ligne — c'est justement le format sous lequel le DOI
    # apparaît en en-tête de PDF, et le supprimer ici l'empêchait d'être
    # indexé (donc invisible au modèle ET au fallback regex en aval).
    lines = [
        l for l in text.split('\n')
        if not (re.match(r'^\s*https?://\S+\s*$', l) and 'doi.org' not in l.lower())
    ]
    return '\n'.join(lines)

# Extraction robuste d'un DOI depuis un texte
def _extract_doi_from_text(text: str):
    """
    Cherche un DOI (URL complète ou forme nue 'doi:10.xxxx/...') dans `text`.

    Approche en 2 temps :
    1) Recherche DIRECTE, sans toucher aux sauts de ligne : "\\S+" s'arrête
       déjà naturellement aussi bien à un saut de ligne qu'à une espace
       normale. Donc si le DOI est suivi d'un espace puis d'autre texte sur
       la même ligne visuelle (ex. "doi:10.3390/su8030198 www.mdpi.com"),
       le résultat est immédiatement correct et borné.
    2) Si le suffixe capturé s'arrête PILE après un "/" (ou est vide) —
       signe que la coupure de justification tombe au milieu du numéro
       lui-même (ex. ".../10.3390/" + saut de ligne + "su17167412") — on
       complète avec UNIQUEMENT le premier mot de la ligne suivante, jamais
       plus, pour ne pas avaler un paragraphe entier qui suivrait
       (ex. un bloc "Copyright: ..." juste en dessous).
    Retourne l'URL complète (https://doi.org/...) ou None.
    """
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
        return None  # toujours incomplet malgré la tentative de complétion

    return "https://doi.org/" + suffix.rstrip(".,);]")

# Indexer un article
def index_article(article_id: str, full_text: str, chunk_size=500, overlap=50):
    words  = full_text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

    embeddings = client.embeddings.create(
        input=chunks,
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    ).data

    collection.add(
        ids        = [f"{article_id}_c{i}" for i in range(len(chunks))],
        embeddings = [e.embedding for e in embeddings],
        documents  = chunks,
        metadatas  = [{"article_id": article_id} for _ in chunks],
    )
    print(f"  [OK] Indexe : {len(chunks)} chunks")

# Récupérer les passages pertinents
def retrieve(article_id: str, query: str, n=10) -> list[str]:
    q_emb = client.embeddings.create(
        input=[query],
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    ).data[0].embedding

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=n,
        where={"article_id": article_id},
    )
    return results["documents"][0]

# Fallback : demander un champ précis si "not reported"
def fallback_field(field_key: str, article_id: str) -> str:
    """
    Appel ciblé pour récupérer un champ critique manquant.
    Interroge GPT avec les premiers chunks + une question directe.
    """
    question = CRITICAL_FIELDS[field_key]

    # Toujours utiliser les premiers chunks pour les métadonnées
    header = collection.get(
        where={"article_id": article_id},
        limit=4,
        include=["documents"],
    )
    context = "\n\n---\n\n".join(header["documents"][:4])

    prompt = (
        f"Here are the first passages of a research article:\n\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer with ONLY the value, nothing else. "
        f"If truly not found, reply: not reported"
    )

    try:
        response = client.chat.completions.create(
            model            = MODEL,
            messages         = [{"role": "user", "content": prompt}],
            reasoning_effort = "medium",
            max_completion_tokens = 100,
        )
    except Exception as e:
        err_msg = str(e).lower()
        if "insufficient_quota" in err_msg or "billing" in err_msg or \
           "exceeded" in err_msg or "quota" in err_msg:
            raise QuotaExhaustedError(f"Crédits API épuisés : {e}")
        raise
    return response.choices[0].message.content.strip()

# Extraire un article
def extract_article(article_id: str, few_shot_examples: list = None, raw_text: str = None,
                     full_text: str = None, doi_from_links: str = None) -> dict:
    import re as _re

    # 0. DOI — ordre de priorité :
    #    a) lien hypertexte lu directement dans le PDF (le plus fiable, ne
    #       dépend pas du rendu visuel du texte)
    #    b) regex sur le texte brut (avant nettoyage/indexation)
    #    c) regex sur les chunks déjà indexés (dernier recours)
    _doi_prefill = doi_from_links

    if not _doi_prefill:
        try:
            _source_text = raw_text
            if _source_text is None:
                _raw_docs = collection.get(
                    where={"article_id": article_id}, limit=20, include=["documents"]
                )
                _source_text = " ".join(_raw_docs.get("documents") or [])

            # Exclure la bibliographie : elle contient les DOI des références
            # citées, pas celui de l'article lui-même.
            _source_text = _re.split(
                r"\n\s*(References|Bibliography|REFERENCES|Références)\s*\n",
                _source_text
            )[0]

            _doi_prefill = _extract_doi_from_text(_source_text)
        except Exception:
            pass

    # 1. Le traitement est séquentiel (un PDF à la fois) : au lieu de
    #    reconstituer le contexte à partir de chunks RAG sélectionnés — qui
    #    peuvent omettre des passages pertinents selon la similarité
    #    sémantique — on donne le texte intégral de l'article au modèle.
    #    Le RAG reste en secours uniquement si l'article dépasse la marge
    #    de sécurité fixée ci-dessous (article inhabituellement long).
    MAX_WORDS_FULL_TEXT = 20000  # large marge pour un article MDPI (~6-9k mots)

    if full_text and len(full_text.split()) <= MAX_WORDS_FULL_TEXT:
        passages = full_text
    else:
        # Premiers chunks (titre, auteur, abstract) — toujours inclus
        header = collection.get(
            where={"article_id": article_id},
            limit=4,
            include=["documents"],
        )
        header_chunks = header["documents"][:4] if header["documents"] else []

        # Chunks sémantiques (méthodes, résultats, équité, gouvernance)
        query = (
            "participatory methods greening intervention design implementation "
            "participants equity justice outcomes governance"
        )
        semantic_chunks = retrieve(article_id, query, n=8)

        # Fusionner sans doublons (header en premier)
        seen, all_chunks = set(), []
        for chunk in header_chunks + semantic_chunks:
            if chunk not in seen:
                seen.add(chunk)
                all_chunks.append(chunk)

        passages = build_passages(all_chunks)

    messages = generate_messages(passages, few_shot_examples or [])

    try:
        response = client.chat.completions.create(
            model            = MODEL,
            messages         = messages,
            reasoning_effort = "medium",
            response_format  = {"type": "json_object"},
        )
    except Exception as e:
        err_msg = str(e).lower()
        if "insufficient_quota" in err_msg or "billing" in err_msg or \
           "exceeded" in err_msg or "quota" in err_msg:
            raise QuotaExhaustedError(f"Crédits API épuisés : {e}")
        raise  # autre erreur -> propager normalement

    result = json.loads(response.choices[0].message.content)
    result["_article_id"] = article_id

    # Injecter le DOI extrait par regex si le modèle ne l'a pas trouvé
    if _doi_prefill:
        current_a1 = str(result.get("A1", "")).strip().lower()
        if current_a1 in ("not reported", "not found", "", "null", "none") or            not current_a1.startswith("http"):
            result["A1"] = _doi_prefill
            print(f"  [DOI] DOI injecte : {_doi_prefill}")

    # 4. Fallback sur les champs critiques manquants
    for field_key, _ in CRITICAL_FIELDS.items():
        current_val = str(result.get(field_key, "")).lower().strip()
        if current_val in ("not reported", "not found", "", "null", "none"):
            print(f"  [INFO] Fallback pour {field_key}...")
            recovered = fallback_field(field_key, article_id)
            print(f"     -> {recovered}")
            result[field_key] = recovered

    return result

# Convertir le resultat JSON en ligne de template

def to_template_row(result: dict) -> dict:
    row = {col: "not reported" for col in TEMPLATE_COLUMNS}
    for model_key, template_col in KEY_MAP.items():
        val = result.get(model_key, "not reported")
        if isinstance(val, list):
            val = "; ".join(str(v) for v in val)
        if val is None:
            val = "not reported"
        row[template_col] = str(val)
    return row

# Écrire le CSV avec les 3 lignes d'en-tête du template
def write_template_csv(rows: list[dict], output_path: str):
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow(
            ["EXTRACTION TEMPLATE — Participatory Methods in Urban Greening Interventions · v2.0"]
            + [""] * (len(TEMPLATE_COLUMNS) - 1)
        )

        section_row, last_section = [], None
        for col in TEMPLATE_COLUMNS:
            sec = SECTION_MAP.get(col, "")
            section_row.append(sec if sec != last_section else "")
            last_section = sec
        writer.writerow(section_row)

        writer.writerow(TEMPLATE_COLUMNS)

        for row in rows:
            writer.writerow([row.get(col, "not reported") for col in TEMPLATE_COLUMNS])

    print(f"[OK] CSV ecrit : {output_path} ({len(rows)} articles)")

# Charger les article_id déjà extraits dans le CSV existant
def _load_done_ids(output_path: str) -> set[str]:
    """
    Lit le CSV de sortie (s'il existe) et retourne l'ensemble des article_id
    déjà traités, identifiés par la colonne 'A2+A3 Author(s) and year'
    contenant une valeur non vide et différente de 'not reported'.
    On utilise le stem du fichier encodé dans la première colonne DOI/URL
    comme clé, mais comme il n'y a pas de colonne article_id dans le template,
    on déduit l'identité via la colonne A1 (DOI) — si elle est remplie,
    l'article a été traité.
    """
    done = set()
    path = Path(output_path)
    if not path.exists():
        return done

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header_rows = [next(reader, None) for _ in range(3)]  # skip 3 header rows
        col_names = header_rows[2] if header_rows[2] else []

        # Trouver l'index de la colonne article_id (ajoutée par append_row)
        if "_article_id" in col_names:
            idx = col_names.index("_article_id")
            for row in reader:
                if len(row) > idx and row[idx].strip():
                    done.add(row[idx].strip())
        # Fallback : pas de colonne _article_id (ancien CSV)
        else:
            # On ne peut pas skip de facon fiable, on retourne vide
            pass

    return done

# Initialiser le CSV avec les en-têtes (si fichier n'existe pas)
def _init_csv_if_needed(output_path: str):
    """Crée le fichier CSV avec les 3 lignes d'en-tête s'il n'existe pas."""
    path = Path(output_path)
    if path.exists() and path.stat().st_size > 0:
        return  # déjà initialisé

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow(
            ["EXTRACTION TEMPLATE — Participatory Methods in Urban Greening Interventions · v2.0"]
            + [""] * (len(TEMPLATE_COLUMNS))  # +1 pour _article_id
        )

        section_row, last_section = [], None
        for col in TEMPLATE_COLUMNS:
            sec = SECTION_MAP.get(col, "")
            section_row.append(sec if sec != last_section else "")
            last_section = sec
        section_row.append("")  # _article_id
        writer.writerow(section_row)

        writer.writerow(TEMPLATE_COLUMNS + ["_article_id"])

# Ajouter une ligne au CSV (écriture incrémentale)
def _append_row(output_path: str, row: dict, article_id: str):
    """Ajoute une ligne de données au CSV existant (mode append)."""
    with open(output_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [row.get(col, "not reported") for col in TEMPLATE_COLUMNS] + [article_id]
        )

# Pipeline complet
def run_pipeline(full_texts_dir: str, output_path: str, few_shot_examples=None):
    import fitz

    full_texts_dir = Path(full_texts_dir)

    # Checkpoint : charger les articles déjà traités
    _init_csv_if_needed(output_path)
    done_ids = _load_done_ids(output_path)
    if done_ids:
        print(f"[INFO] {len(done_ids)} articles deja extraits, ils seront ignores")

    files = sorted(
        list(full_texts_dir.glob("*.pdf")) +
        list(full_texts_dir.glob("*.txt"))
    )
    print(f"[INFO] {len(files)} fichiers trouves dans {full_texts_dir}")

    n_extracted = 0
    n_skipped   = 0

    for f in files:
        article_id = f.stem

        # Skip si déjà traité
        if article_id in done_ids:
            n_skipped += 1
            continue

        print(f"\n-> {article_id}")

        doi_from_links = None
        doi_own_text   = None

        if f.suffix == ".pdf":
            doc  = fitz.open(str(f))
            text = "\n".join([p.get_text() for p in doc])

            # PRIORITÉ 1 : le DOI PROPRE à l'article est presque toujours
            # imprimé en clair dans le bandeau de citation de la page 1
            import re as _re_doi
            _first_pages_text = "".join(p.get_text() for p in doc[:2])
            doi_own_text = _extract_doi_from_text(_first_pages_text)

            # PRIORITÉ 2 (secours uniquement) : lien hypertexte cliquable
            if not doi_own_text:
                try:
                    for page in doc[:2]:
                        for link in page.get_links():
                            uri = link.get("uri", "") or ""
                            if "doi.org" in uri.lower():
                                doi_from_links = uri.strip().rstrip(".,);]")
                                break
                        if doi_from_links:
                            break
                except Exception:
                    pass

            # Diagnostic inline (toujours affiché)
            print(f"  [DOI] DOI trouve dans le texte (p.1-2) : {doi_own_text or '(aucun)'}")
            if not doi_own_text:
                print(f"  [DOI] DOI via lien PDF (p.1-2)        : {doi_from_links or '(aucun)'}")
            if not doi_own_text and not doi_from_links:
                _n_links = sum(len(p.get_links()) for p in doc[:2])
                print(f"     Liens totaux dans les 2 premières pages : {_n_links}")
                _page1_sample = doc[0].get_text()[:300].replace("\n", " ⏎ ")
                print(f"     Aperçu texte page 1 : {_page1_sample!r}")
        else:
            text = f.read_text(encoding="utf-8", errors="ignore")

        doi_prefill_from_pipeline = doi_own_text or doi_from_links

        raw_text_for_doi = text  # conservé avant nettoyage
        full_text_clean  = clean_text(text)

        if len(full_text_clean.split()) < 200:
            print(f"  [ATTENTION] Texte trop court ({len(full_text_clean.split())} mots), ignore")
            continue

        # Réindexer avec texte nettoyé
        existing = collection.get(where={"article_id": article_id}, limit=1)
        if existing["ids"]:
            collection.delete(where={"article_id": article_id})
            print(f"  [INFO] Ancienne version supprimee")
        index_article(article_id, full_text_clean)

        try:
            result = extract_article(
                article_id, few_shot_examples,
                raw_text=raw_text_for_doi,
                full_text=full_text_clean,
                doi_from_links=doi_prefill_from_pipeline,
            )
            row = to_template_row(result)

            # Écriture incrémentale : chaque article sauvé immédiatement
            _append_row(output_path, row, article_id)
            n_extracted += 1
            print(f"  [OK] Extrait et sauvegarde ({n_extracted} total)")

        except QuotaExhaustedError as e:
            print(f"\n[ARRET] {e}")
            print(f"   {n_extracted} articles extraits cette session.")
            print(f"   Recharge tes crédits puis relance le script — il reprendra automatiquement.")
            return

        except Exception as e:
            print(f"  [ERREUR] {e}")

    print(f"\n[FIN] Pipeline termine : {n_extracted} nouveaux + {n_skipped} deja faits")

# Point d'entrée
if __name__ == "__main__":
    run_pipeline(
        full_texts_dir = "./full_texts_pdf",
        output_path    = "./extraction/resultats_extraction.csv",
    )

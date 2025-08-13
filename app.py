import os
import time
import logging
import requests
import openai
import psycopg2
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("scanapi")

# ---------------- Environment ----------------
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PERPLEXITY_API_KEY = os.environ["PERPLEXITY_API_KEY"]
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4-turbo")

# Database
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
PGSSLMODE = os.environ.get("PGSSLMODE", "require")  # gebruik "disable" voor interne Render-DB

# Tuning (kan via Render env worden overschreven)
DEFAULT_MAX_N            = int(os.environ.get("MAX_QUESTIONS", "10"))     # max aantal vragen
MAX_SCAN_SECONDS         = int(os.environ.get("MAX_SCAN_SECONDS", "300")) # totaal budget (5 min)
PERPLEXITY_HTTP_TIMEOUT  = float(os.environ.get("PERPLEXITY_TIMEOUT", "25"))  # read-timeout per call
SLEEP_FAST               = float(os.environ.get("SLEEP_FAST", "0.5"))     # kleine pauze tussen calls

# Toegestane origins voor CORS (komma-gescheiden)
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "https://aseo-70fee3.webflow.io").split(",")
    if o.strip()
]

# OpenAI client
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)

# CORS (globaal + specifiek /scan)
CORS(app, resources={
    r"/scan": {
        "origins": ALLOWED_ORIGINS or ["*"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
    },
    r"/ping": {"origins": "*"},
})

# ---------- DB helper ----------
def save_scan_to_db(name: str, website_url: str | None, description: str | None,
                    location: str | None, language: str | None, score: int) -> bool:
    """
    Slaat één rij op in tabel 'scans'. Returnt True bij succes, False bij skip/fout.
    Vereist env: DATABASE_URL (postgresql://user:pass@host/db).
    """
    if not DATABASE_URL:
        log.info("DATABASE_URL ontbreekt; sla DB-write over.")
        return False
    try:
        # psycopg2 accepteert direct de URL; sslmode kan je via PGSSLMODE sturen.
        with psycopg2.connect(DATABASE_URL, sslmode=PGSSLMODE) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO scans (name, website_url, description, location, language, score)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (name, website_url or None, description or None, location or None, language or None, int(score)),
                )
        return True
    except Exception as e:
        log.exception("DB insert failed: %s", e)
        return False

# --------------- Helpers ----------------
def genereer_zoekvragen(description: str, locatie: str, n: int = 10, language: str | None = None):
    """Maak n natuurlijke AI-zoekvragen op basis van de beschrijving."""
    try:
        n = int(n)
    except Exception:
        n = DEFAULT_MAX_N
    n = max(1, min(n, DEFAULT_MAX_N))

    lang_hint = ""
    if language:
        lang_hint = f"\nSchrijf de vragen in het **{language}**."

    prompt = f"""
Je bent een SEO-expert gespecialiseerd in AI-zoekgedrag.

Je krijgt hieronder een omschrijving van een bedrijf/dienst.

1) Bepaal impliciet de kern:
   - business type
   - 2–3 belangrijkste diensten/producten
   - eventuele unieke kenmerken
2) Genereer vervolgens {n} natuurlijke vragen die een gebruiker aan AI-zoekmachines
   (zoals ChatGPT of Perplexity) zou stellen om zo'n aanbieder in **{locatie}** te vinden.
   - Realistisch, kort en natuurlijk geformuleerd.
   - Focus op zoeken/vergelijken van aanbieders.
   - Geen bedrijfsnamen, fictieve situaties of uitleg.
   - Eén vraag per regel.{lang_hint}

Omschrijving:
\"\"\"{description}\"\"\"
""".strip()

    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Je bent een expert in gebruikerszoekgedrag en SEO."},
                {"role": "user", "content": prompt},
            ],
        )
        content = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        log.exception("OpenAI-fout bij genereren vragen: %s", e)
        return []

    regels = [r.strip("-• ").strip() for r in content.split("\n") if r.strip()]
    if len(regels) > n:
        met_vraagteken = [r for r in regels if "?" in r]
        regels = (met_vraagteken or regels)[:n]
    return regels


def vraag_perplexity(prompt: str, return_errors: bool = False):
    """
    Vraagt Perplexity. Bij succes: antwoordtekst.
    Bij fout: None (of als return_errors=True een korte foutstring, zichtbaar in 'items').
    """
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "sonar",  # evt. 'sonar-small-chat' voor nog iets sneller/goedkoper
        "messages": [{
            "role": "user",
            "content": (
                "Beantwoord de volgende vraag kort en concreet, in maximaal 3 zinnen. "
                "Noem alleen bedrijven, merknamen, locaties of domeinen. Geen uitleg.\n\n" + prompt
            )
        }],
    }
    try:
        # tuple timeout: (connect_timeout, read_timeout)
        r = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload,
            timeout=(10, PERPLEXITY_HTTP_TIMEOUT),
        )
    except requests.Timeout:
        log.warning("Perplexity timeout")
        return "__ERR timeout" if return_errors else None
    except requests.RequestException as e:
        log.warning("Perplexity netwerkfout: %s", e)
        return (f"__ERR network: {e}") if return_errors else None

    if r.status_code != 200:
        snippet = r.text[:180].replace("\n", " ")
        log.info("Perplexity %s: %s", r.status_code, snippet)
        return (f"__ERR {r.status_code}: {snippet}") if return_errors else None

    try:
        return r.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as e:
        log.warning("Perplexity parse-fout: %s; body=%s", e, r.text[:200])
        return (f"__ERR parse: {e}") if return_errors else None


def check_bedrijfsvermelding(antwoord: str, bedrijfsnaam: str, domeinnaam: str | None = None) -> bool:
    if not antwoord:
        return False
    t = antwoord.lower()
    return (bedrijfsnaam and bedrijfsnaam.lower() in t) or (domeinnaam and domeinnaam.lower() in t)


def run_vindbaarheidsscan(
    bedrijfsnaam: str,
    description: str,
    locatie: str,
    domeinnaam: str | None,
    n: int = 10,
    collect: bool = False,
    language: str | None = None,
):
    """Als collect=True, retourneer (score, items) met Q&A."""
    try:
        n = int(n)
    except Exception:
        n = DEFAULT_MAX_N
    n = max(1, min(n, DEFAULT_MAX_N))

    start_ts = time.time()

    vragen = genereer_zoekvragen(description, locatie, n=n, language=language)
    if not vragen:
        return 0 if not collect else (0, [])

    hits = 0
    items = []
    processed = 0

    for vraag in vragen:
        # Respecteer totaalbudget
        if time.time() - start_ts > MAX_SCAN_SECONDS:
            log.info("Time budget reached; stopping early after %d/%d vragen", processed, len(vragen))
            break

        antw = vraag_perplexity(vraag, return_errors=collect)
        processed += 1

        hit = bool(antw and check_bedrijfsvermelding(antw, bedrijfsnaam, domeinnaam))
        if hit:
            hits += 1
        if collect:
            items.append({"q": vraag, "a": (antw or ""), "hit": hit})

        time.sleep(SLEEP_FAST)  # kleine pauze i.v.m. rate-limits

    total = max(processed, 1)
    score = round((hits / total) * 100)
    return score if not collect else (score, items)


# --------------- API ----------------
@app.route("/", methods=["GET"])
def root():
    return "scanner up", 200


@app.route("/ping", methods=["GET"])
def ping():
    return "ok", 200


@app.route("/scan", methods=["POST", "OPTIONS"])
@cross_origin(
    origins=ALLOWED_ORIGINS or ["*"],
    methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)
def scan():
    # Preflight (CORS)
    if request.method == "OPTIONS":
        return ("", 204)

    # JSON of form-encoded accepteren
    data = (request.get_json(silent=True) or request.form.to_dict() or {})
    log.info("DEBUG /scan incoming: %s", data)

    bedrijfsnaam = (data.get("company_name") or "").strip()
    description  = (data.get("description")  or "").strip()
    locatie      = (data.get("location")    or "").strip()
    website_url  = (data.get("website_url") or "").strip()
    language     = (data.get("language")    or "").strip() or None

    domein = (
        website_url.replace("https://", "").replace("http://", "").replace("/", "").lower()
    ) if website_url else None

    try:
        n = int(data.get("n", DEFAULT_MAX_N))
    except (TypeError, ValueError):
        n = DEFAULT_MAX_N
    n = max(1, min(n, DEFAULT_MAX_N))

    return_details = bool(data.get("return_details") or data.get("debug"))

    # Validate
    missing = []
    if not bedrijfsnaam: missing.append("company_name")
    if not description:  missing.append("description")
    if not locatie:      missing.append("location")
    if missing:
        return jsonify({
            "error": "missing required fields",
            "missing": missing,
            "received": {
                "company_name": bedrijfsnaam,
                "description": description,
                "location": locatie,
                "website_url": website_url
            }
        }), 400

    if return_details:
        score, items = run_vindbaarheidsscan(
            bedrijfsnaam, description, locatie, domein, n=n, collect=True, language=language
        )
        # Opslaan in DB
        save_scan_to_db(bedrijfsnaam, website_url, description, locatie, language, score)
        return jsonify({"score": score, "items": items}), 200
    else:
        score = run_vindbaarheidsscan(
            bedrijfsnaam, description, locatie, domein, n=n, collect=False, language=language
        )
        # Opslaan in DB
        save_scan_to_db(bedrijfsnaam, website_url, description, locatie, language, score)
        return jsonify({"score": score}), 200


if __name__ == "__main__":
    # Local dev — Render gebruikt gunicorn in productie
    app.run(host="0.0.0.0", port=5000)

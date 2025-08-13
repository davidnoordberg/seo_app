import os
import time
import logging
import re
import requests
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from sqlalchemy import create_engine, text

# ---------------- Logging ----------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
log = logging.getLogger("scanapi")

# ---------------- Environment ----------------
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PERPLEXITY_API_KEY = os.environ["PERPLEXITY_API_KEY"]
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4-turbo")

# Database (Render Internal/External URL)
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

def _adapt_url_for_sqlalchemy(url: str) -> str:
    """
    Turn 'postgresql://user:pass@host/db' -> 'postgresql+pg8000://user:pass@host/db'
    (and 'postgres://...' -> 'postgresql+pg8000://...')
    """
    if not url:
        return ""
    if url.startswith("postgres://"):
        url = "postgresql://" + url.split("://", 1)[1]
    if url.startswith("postgresql://"):
        url = "postgresql+pg8000://" + url.split("://", 1)[1]
    return url

ENGINE = create_engine(_adapt_url_for_sqlalchemy(DATABASE_URL), pool_pre_ping=True) if DATABASE_URL else None

# Tuning (overridable via env)
DEFAULT_MAX_N            = int(os.environ.get("MAX_QUESTIONS", "10"))      # max questions
MAX_SCAN_SECONDS         = int(os.environ.get("MAX_SCAN_SECONDS", "300"))  # total budget (5 min)
PERPLEXITY_HTTP_TIMEOUT  = float(os.environ.get("PERPLEXITY_TIMEOUT", "25"))
SLEEP_FAST               = float(os.environ.get("SLEEP_FAST", "0.5"))

# CORS
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "https://aseo-70fee3.webflow.io").split(",")
    if o.strip()
]

# OpenAI client
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)

# CORS (global + explicit /scan & /scans)
CORS(app, resources={
    r"/scan": {
        "origins": ALLOWED_ORIGINS or ["*"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
    },
    r"/scans": {"origins": ALLOWED_ORIGINS or ["*"], "methods": ["GET"]},
    r"/ping": {"origins": "*"},
})

# ---------- DB helper ----------
def save_scan_to_db(name: str, website_url: str | None, description: str | None,
                    location: str | None, language: str | None, score: int,
                    email: str | None) -> bool:
    """
    Insert one row into 'scans' table. Returns True on success.
    """
    if not ENGINE:
        log.info("DATABASE_URL missing / engine not configured; skipping DB write.")
        return False
    try:
        with ENGINE.begin() as conn:
            conn.execute(text("""
                INSERT INTO scans (name, website_url, description, location, language, score, email)
                VALUES (:name, :website_url, :description, :location, :language, :score, :email)
            """), dict(
                name=name,
                website_url=website_url or None,
                description=description or None,
                location=location or None,
                language=language or None,
                score=int(score),
                email=email or None
            ))
        return True
    except Exception as e:
        log.exception("DB insert failed: %s", e)
        return False

# ---------------- Parse & validate helpers ----------------
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

def _normalize_url(url: str | None) -> str | None:
    if not url:
        return None
    u = url.strip()
    if not u:
        return None
    if not u.lower().startswith(("http://", "https://")):
        u = "https://" + u
    return u

def parse_scan_request(req) -> dict:
    """
    Accept JSON or form-encoded bodies and tolerate common Webflow name variants.
    """
    data = (req.get_json(silent=True) or {})
    if not data and req.form:
        data = req.form.to_dict(flat=True)

    def g(key, *alts, default=""):
        for k in (key, *alts):
            if k in data:
                v = data.get(k)
                if isinstance(v, str):
                    v = v.strip()
                return v
        return default

    payload = {
        "company_name": g("company_name"),
        "website_url":  _normalize_url(g("website_url")),
        "description":  g("description"),
        "location":     g("location"),
        "language":     (g("language") or "en").lower(),
        # Be forgiving about field names coming from Webflow
        "email":        g("email", "Email", "email_address", "Email-2"),
        "n":            int(g("n", default=DEFAULT_MAX_N) or DEFAULT_MAX_N),
        "return_details": str(g("return_details", "debug", default="")).lower() in ("1","true","yes","on")
    }

    # Clamp n
    payload["n"] = max(1, min(payload["n"], DEFAULT_MAX_N))
    return payload

# --------------- Scan logic ----------------
def genereer_zoekvragen(description: str, locatie: str, n: int = 10, language: str | None = None):
    """Create n natural AI-search questions based on a short business description."""
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
        log.exception("OpenAI error while generating questions: %s", e)
        return []

    regels = [r.strip("-• ").strip() for r in content.split("\n") if r.strip()]
    if len(regels) > n:
        met_vraagteken = [r for r in regels if "?" in r]
        regels = (met_vraagteken or regels)[:n]
    return regels

def vraag_perplexity(prompt: str, return_errors: bool = False):
    """
    Call Perplexity. On success: answer text.
    On failure: None (or error marker when return_errors=True).
    """
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "sonar",
        "messages": [{
            "role": "user",
            "content": (
                "Beantwoord de volgende vraag kort en concreet, in maximaal 3 zinnen. "
                "Noem alleen bedrijven, merknamen, locaties of domeinen. Geen uitleg.\n\n" + prompt
            )
        }],
    }
    try:
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
        log.warning("Perplexity network error: %s", e)
        return (f"__ERR network: {e}") if return_errors else None

    if r.status_code != 200:
        snippet = r.text[:180].replace("\n", " ")
        log.info("Perplexity %s: %s", r.status_code, snippet)
        return (f"__ERR {r.status_code}: {snippet}") if return_errors else None

    try:
        return r.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as e:
        log.warning("Perplexity parse error: %s; body=%s", e, r.text[:200])
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
    """If collect=True, return (score, items) with Q&A details."""
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
        if time.time() - start_ts > MAX_SCAN_SECONDS:
            log.info("Time budget reached; stopping early after %d/%d questions", processed, len(vragen))
            break

        antw = vraag_perplexity(vraag, return_errors=collect)
        processed += 1

        hit = bool(antw and check_bedrijfsvermelding(antw, bedrijfsnaam, domeinnaam))
        if hit:
            hits += 1
        if collect:
            items.append({"q": vraag, "a": (antw or ""), "hit": hit})

        time.sleep(SLEEP_FAST)

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

    # Deep debug (enable with LOG_LEVEL=DEBUG)
    log.debug("HEADERS=%s", dict(request.headers))
    log.debug("RAW=%s | FORM=%s | JSON=%s",
              request.get_data(as_text=True),
              request.form.to_dict(),
              request.get_json(silent=True))

    payload = parse_scan_request(request)
    log.info("DEBUG /scan incoming (normalized): %s", payload)

    # Validate with clear messages
    missing = [k for k in ("company_name", "description", "location", "email") if not payload.get(k)]
    if missing:
        return jsonify({"error": f"Missing required field(s): {', '.join(missing)}"}), 400

    if not EMAIL_RE.match(payload["email"]):
        return jsonify({"error": "Invalid email format"}), 400

    domein = None
    if payload["website_url"]:
        domein = payload["website_url"].replace("https://", "").replace("http://", "").split("/")[0].lower()

    if payload["return_details"]:
        score, items = run_vindbaarheidsscan(
            payload["company_name"], payload["description"], payload["location"],
            domein, n=payload["n"], collect=True, language=payload["language"]
        )
        save_scan_to_db(payload["company_name"], payload["website_url"], payload["description"],
                        payload["location"], payload["language"], score, payload["email"])
        return jsonify({"score": score, "items": items}), 200
    else:
        score = run_vindbaarheidsscan(
            payload["company_name"], payload["description"], payload["location"],
            domein, n=payload["n"], collect=False, language=payload["language"]
        )
        save_scan_to_db(payload["company_name"], payload["website_url"], payload["description"],
                        payload["location"], payload["language"], score, payload["email"])
        return jsonify({"score": score}), 200

@app.route("/scans", methods=["GET"])
def list_scans():
    """Return all scans as JSON."""
    if not ENGINE:
        return jsonify({"error": "Database not configured"}), 500
    try:
        with ENGINE.connect() as conn:
            result = conn.execute(text(
                "SELECT id, created_at, name, website_url, description, location, language, score, email "
                "FROM scans ORDER BY created_at DESC"
            ))
            rows = [dict(r) for r in result.mappings()]
        return jsonify(rows), 200
    except Exception as e:
        log.exception("DB query failed: %s", e)
        return jsonify({"error": "DB query failed"}), 500

if __name__ == "__main__":
    # Local dev — Render uses gunicorn in production
    app.run(host="0.0.0.0", port=5000)

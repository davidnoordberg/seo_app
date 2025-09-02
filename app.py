import os
import time
import logging
import requests
import openai
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from sqlalchemy import create_engine, text

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("scanapi")

# ---------------- Environment ----------------
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PERPLEXITY_API_KEY = os.environ["PERPLEXITY_API_KEY"]
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4-turbo")

# reCAPTCHA secret
RECAPTCHA_SECRET = os.environ.get("RECAPTCHA_SECRET", "").strip()

# Database (gebruik je Internal URL op Render als DATABASE_URL)
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

# SMTP / contact
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.zoho.eu")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "info@aseon.io")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
CONTACT_TO = os.environ.get("CONTACT_TO", "info@aseon.io")

def _adapt_url_for_sqlalchemy(url: str) -> str:
    """
    Maak van 'postgresql://user:pass@host/db' -> 'postgresql+pg8000://user:pass@host/db'
    (of 'postgres://' -> 'postgresql+pg8000://...')
    """
    if not url:
        return ""
    if url.startswith("postgres://"):
        url = "postgresql://" + url.split("://", 1)[1]
    if url.startswith("postgresql://"):
        url = "postgresql+pg8000://" + url.split("://", 1)[1]
    return url

ENGINE = create_engine(_adapt_url_for_sqlalchemy(DATABASE_URL), pool_pre_ping=True) if DATABASE_URL else None

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

# CORS (globaal + specifiek /scan en /contact)
CORS(app, resources={
    r"/scan": {
        "origins": ALLOWED_ORIGINS or ["*"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
    },
    r"/contact": {
        "origins": ALLOWED_ORIGINS or ["*"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
    },
    r"/ping": {"origins": "*"},
})

# ---------- DB helper ----------
def save_scan_to_db(name: str, website_url: str | None, description: str | None,
                    location: str | None, language: str | None, score: int,
                    email: str | None) -> bool:
    """
    Slaat één rij op in tabel 'scans'. Returnt True bij succes, False bij skip/fout.
    Vereist env: DATABASE_URL (Internal of External). Tabel 'scans' heb je al aangemaakt.
    """
    if not ENGINE:
        log.info("DATABASE_URL ontbreekt of engine niet geconfigureerd; sla DB-write over.")
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

# --------------- reCAPTCHA helper ---------------
def verify_recaptcha(token: str, remote_ip: str | None = None):
    """Valideer reCAPTCHA v2 token server-side. Return (ok: bool, details: dict/str)"""
    if not RECAPTCHA_SECRET:
        return False, {"error": "server_misconfigured: missing RECAPTCHA_SECRET"}
    try:
        r = requests.post(
            "https://www.google.com/recaptcha/api/siteverify",
            data={
                "secret": RECAPTCHA_SECRET,
                "response": token,
                "remoteip": remote_ip or "",
            },
            timeout=10,
        )
        j = r.json()
        return bool(j.get("success")), j
    except requests.RequestException as e:
        log.exception("reCAPTCHA verify error: %s", e)
        return False, {"error": "network_error"}

# --------------- Helpers ----------------
def genereer_zoekvragen(description: str, locatie: str, n: int = 10,
                         language: str | None = None, biz_type: str | None = None):
    """
    Genereer n natuurlijke AI-zoekvragen.
      - online  : land/regio-focus, geen stad/‘near me’
      - physical: lokaal zoeken (stad/regio toegestaan)
    """
    try:
        n = int(n)
    except Exception:
        n = DEFAULT_MAX_N
    n = max(1, min(n, DEFAULT_MAX_N))

    lang_hint = f"\nSchrijf de vragen in het **{language}**." if language else ""

    if (biz_type or "").lower() == "online":
        scope_hint = (
            f"\n- Behandel **{locatie}** als primaire markt (land/regio)."
            f"\n- Vermijd plaatsnamen, ‘near me’ en straatadressen."
            f"\n- Focus op vergelijken/kiezen van online aanbieders, levering, prijs/kwaliteit, reviews."
        )
        examples = (
            "\nVoorbeelden (niet kopiëren, alleen als stijl):"
            f"\n• Waar koop ik [product/dienst] online in {locatie}?"
            f"\n• Welke [productcategorie] platforms leveren snel in {locatie}?"
            f"\n• Top beoordeelde aanbieders voor [dienst] in {locatie}?"
            f"\n• Beste prijs/kwaliteit voor [product] met bezorging in {locatie}?"
        )
    else:
        scope_hint = (
            f"\n- Richt je op lokaal zoeken in/om **{locatie}** (plaats-/regiobenamingen zijn goed)."
            f"\n- ‘near me’, openingstijden, afhalen/afspraak mogen."
        )
        examples = (
            "\nVoorbeelden (niet kopiëren, alleen als stijl):"
            f"\n• Beste [dienst] in {locatie}?"
            f"\n• Betaalbare [dienst] nabij {locatie} (open vandaag)?"
            f"\n• Waar kan ik [product] kopen in {locatie}?"
        )

    prompt = f'''
Je bent een SEO-expert in AI-zoekgedrag.

Opdracht:
1) Begrijp kort de kern uit de omschrijving (type aanbieder, topdiensten, unieke punten).
2) Genereer {n} natuurlijke vragen die een gebruiker aan ChatGPT/Perplexity zou stellen
   om zo'n aanbieder te vinden.{scope_hint}
   - Eén vraag per regel, kort en natuurlijk.
   - Geen bedrijfsnamen in de vraag, geen uitleg of nummering.{lang_hint}
{examples}

Omschrijving:
"""{description}"""
'''.strip()

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
    biz_type: str | None = None,
):
    """Als collect=True, retourneer (score, items) met Q&A."""
    try:
        n = int(n)
    except Exception:
        n = DEFAULT_MAX_N
    n = max(1, min(n, DEFAULT_MAX_N))

    start_ts = time.time()

    vragen = genereer_zoekvragen(description, locatie, n=n, language=language, biz_type=biz_type)
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

    # JSON of form-encoded accepteren
    data = (request.get_json(silent=True) or request.form.to_dict() or {})
    log.info("DEBUG /scan incoming: %s", data)

    # ✅ reCAPTCHA: verplicht token + verificatie vóór verdere verwerking
    recaptcha_token = (data.get("recaptcha_token") or "").strip()
    if not recaptcha_token:
        return jsonify({"error": "missing recaptcha_token"}), 400

    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    ok, details = verify_recaptcha(recaptcha_token, client_ip)
    if not ok:
        return jsonify({"error": "failed_recaptcha", "details": details}), 403

    bedrijfsnaam = (data.get("company_name") or "").strip()
    description  = (data.get("description")  or "").strip()
    locatie      = (data.get("location")    or "").strip()
    website_url  = (data.get("website_url") or "").strip()
    email        = (data.get("email")       or "").strip() or None
    language     = (data.get("language")    or "").strip() or None
    biz_type     = (data.get("biz_type")    or "").strip().lower() or None  # "online" | "physical"

    domein = (
        website_url.replace("https://", "").replace("http://", "").replace("/", "").lower()
    ) if website_url else None

    try:
        n = int(data.get("n", DEFAULT_MAX_N))
    except (TypeError, ValueError):
        n = DEFAULT_MAX_N
    n = max(1, min(n, DEFAULT_MAX_N))

    return_details = bool(data.get("return_details") or data.get("debug"))

    # Validate overige velden
    missing = []
    if not bedrijfsnaam: missing.append("company_name")
    if not description:  missing.append("description")
    if not locatie:      missing.append("location")
    if not email:        missing.append("email")
    if missing:
        return jsonify({
            "error": "missing required fields",
            "missing": missing,
            "received": {
                "company_name": bedrijfsnaam,
                "description": description,
                "location": locatie,
                "website_url": website_url,
                "email": email,
                "biz_type": biz_type
            }
        }), 400

    if return_details:
        score, items = run_vindbaarheidsscan(
            bedrijfsnaam, description, locatie, domein, n=n, collect=True, language=language, biz_type=biz_type
        )
        save_scan_to_db(bedrijfsnaam, website_url, description, locatie, language, score, email)
        return jsonify({"score": score, "items": items}), 200
    else:
        score = run_vindbaarheidsscan(
            bedrijfsnaam, description, locatie, domein, n=n, collect=False, language=language, biz_type=biz_type
        )
        save_scan_to_db(bedrijfsnaam, website_url, description, locatie, language, score, email)
        return jsonify({"score": score}), 200


@app.route("/scans", methods=["GET"])
def list_scans():
    """Geef alle scans terug als JSON."""
    if not ENGINE:
        return jsonify({"error": "Database niet geconfigureerd"}), 500
    try:
        with ENGINE.connect() as conn:
            result = conn.execute(text(
                "SELECT id, created_at, name, website_url, description, location, language, score, email FROM scans ORDER BY created_at DESC"
            ))
            rows = [dict(r) for r in result.mappings()]
        return jsonify(rows), 200
    except Exception as e:
        log.exception("Fout bij ophalen scans: %s", e)
        return jsonify({"error": "DB query failed"}), 500


# --------------- CONTACT: mail via Zoho SMTP ---------------
def send_mail_via_zoho(subject: str, html_body: str, reply_to: str | None = None):
    if not (SMTP_USER and SMTP_PASS):
        raise RuntimeError("SMTP credentials missing")
    msg = MIMEText(html_body, "html", "utf-8")
    msg["Subject"] = subject
    msg["From"] = formataddr(("ASEO Contact", SMTP_USER))
    msg["To"] = CONTACT_TO
    if reply_to:
        msg["Reply-To"] = reply_to
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.sendmail(SMTP_USER, [CONTACT_TO], msg.as_string())

@app.route("/contact", methods=["POST", "OPTIONS"])
@cross_origin(
    origins=ALLOWED_ORIGINS or ["*"],
    methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)
def contact():
    # Preflight
    if request.method == "OPTIONS":
        return ("", 204)

    data = (request.get_json(silent=True) or {})
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip()
    message = (data.get("message") or "").strip()

    if not (name and email and message):
        return jsonify({"error": "missing fields"}), 400

    # (Optioneel) reCAPTCHA voor contact:
    # token = (data.get("recaptcha_token") or "").strip()
    # if not token: return jsonify({"error": "missing recaptcha_token"}), 400
    # ok,_ = verify_recaptcha(token, request.headers.get("X-Forwarded-For", request.remote_addr))
    # if not ok: return jsonify({"error": "failed_recaptcha"}), 403

    subj = f"New contact form message from {name}"
    body = f"""
      <h3>New contact message</h3>
      <p><strong>Name:</strong> {name}<br>
         <strong>Email:</strong> {email}</p>
      <p style="white-space:pre-wrap">{message}</p>
      <hr><p>Sent from aseon.io/contact</p>
    """

    try:
        send_mail_via_zoho(subj, body, reply_to=email)
        return jsonify({"ok": True}), 200
    except Exception as e:
        log.exception("contact send failed: %s", e)
        return jsonify({"error": "email_failed"}), 500


if __name__ == "__main__":
    # Local dev — Render gebruikt gunicorn in productie
    app.run(host="0.0.0.0", port=5000)

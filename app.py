import os
import time
import logging
import requests
import openai
import smtplib
import stripe
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

# Database (Render Internal URL als DATABASE_URL)
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

# SMTP / contact
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.zoho.eu")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "info@aseon.io")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
CONTACT_TO = os.environ.get("CONTACT_TO", "info@aseon.io")

# Stripe
STRIPE_SECRET = os.environ.get("STRIPE_SECRET", "").strip()
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "").strip()
if STRIPE_SECRET:
    stripe.api_key = STRIPE_SECRET

# Checkout redirect urls
CHECKOUT_SUCCESS_URL = os.environ.get(
    "CHECKOUT_SUCCESS_URL",
    "https://aseo-70fee3.webflow.io/thanks?session_id={CHECKOUT_SESSION_ID}",
)
CHECKOUT_CANCEL_URL = os.environ.get(
    "CHECKOUT_CANCEL_URL",
    "https://aseo-70fee3.webflow.io/subscription?canceled=1",
)

def _adapt_url_for_sqlalchemy(url: str) -> str:
    if not url:
        return ""
    if url.startswith("postgres://"):
        url = "postgresql://" + url.split("://", 1)[1]
    if url.startswith("postgresql://"):
        url = "postgresql+pg8000://" + url.split("://", 1)[1]
    return url

ENGINE = create_engine(_adapt_url_for_sqlalchemy(DATABASE_URL), pool_pre_ping=True) if DATABASE_URL else None

# Tuning
DEFAULT_MAX_N            = int(os.environ.get("MAX_QUESTIONS", "10"))
MAX_SCAN_SECONDS         = int(os.environ.get("MAX_SCAN_SECONDS", "300"))
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
CORS(app, resources={
    r"/scan": {"origins": ALLOWED_ORIGINS or ["*"], "methods": ["POST", "OPTIONS"], "allow_headers": ["Content-Type"]},
    r"/contact": {"origins": ALLOWED_ORIGINS or ["*"], "methods": ["POST", "OPTIONS"], "allow_headers": ["Content-Type"]},
    r"/ping": {"origins": "*"},
    r"/checkout/start": {"origins": ALLOWED_ORIGINS or ["*"], "methods": ["POST", "OPTIONS"], "allow_headers": ["Content-Type"]},
    r"/webhook/stripe": {"origins": "*"},
})

# ---------- Stripe plan config (uses your PRODUCT ids + EUR amounts) ----------
PLAN_CONFIG = {
    "basic":    {"product": "prod_SzJLEa6TFnCBvC",   "amount": 17900, "interval": "month", "name": "Basic"},
    "standard": {"product": "prod_SzJMgfOwJCIA44",   "amount": 34900, "interval": "month", "name": "Standard"},
    "premium":  {"product": "prod_SzJNSVZi2ow0od",   "amount": 59900, "interval": "month", "name": "Premium"},
    "boost":    {"product": "prod_SzJOgIxi9e0OLZ",   "amount": 84900, "interval": None,    "name": "Boost"},
}
CURRENCY = "eur"

# ---------- DB helpers ----------
def save_scan_to_db(name: str, website_url: str | None, description: str | None,
                    location: str | None, language: str | None, score: int,
                    email: str | None) -> bool:
    if not ENGINE:
        log.info("DATABASE_URL ontbreekt; sla DB-write over.")
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


def save_payment_event(event_id: str, event_type: str, mode: str | None, status: str | None,
                       is_subscription: bool, customer_email: str | None, customer_name: str | None,
                       plan: str | None, amount_total: int | None, currency: str | None, raw: dict | None):
    if not ENGINE:
        log.info("DATABASE_URL ontbreekt; sla payment DB-write over.")
        return False
    try:
        with ENGINE.begin() as conn:
            conn.execute(text("""
                INSERT INTO payments (event_id, event_type, mode, status, is_subscription,
                                      customer_email, customer_name, plan, amount_total, currency, raw)
                VALUES (:event_id, :event_type, :mode, :status, :is_subscription,
                        :customer_email, :customer_name, :plan, :amount_total, :currency, :raw::jsonb)
                ON CONFLICT (event_id) DO NOTHING
            """), dict(
                event_id=event_id,
                event_type=event_type,
                mode=mode,
                status=status,
                is_subscription=is_subscription,
                customer_email=customer_email,
                customer_name=customer_name,
                plan=plan,
                amount_total=amount_total,
                currency=currency,
                raw=(raw or {})
            ))
        return True
    except Exception as e:
        log.exception("Payment insert failed: %s", e)
        return False


def save_checkout_session(session_id: str, customer_email: str | None, customer_name: str | None,
                          company_name: str | None, competitors: str | None, notes: str | None,
                          plan: str | None, amount_total: int | None, currency: str | None, raw: dict | None):
    if not ENGINE:
        log.info("DATABASE_URL ontbreekt; sla checkout_session DB-write over.")
        return False
    try:
        with ENGINE.begin() as conn:
            conn.execute(text("""
                INSERT INTO checkout_sessions
                  (session_id, customer_email, customer_name, company_name, competitors, notes,
                   plan, amount_total, currency, raw)
                VALUES
                  (:session_id, :customer_email, :customer_name, :company_name, :competitors, :notes,
                   :plan, :amount_total, :currency, :raw::jsonb)
                ON CONFLICT (session_id) DO NOTHING
            """), dict(
                session_id=session_id,
                customer_email=customer_email,
                customer_name=customer_name,
                company_name=company_name,
                competitors=competitors,
                notes=notes,
                plan=plan,
                amount_total=amount_total,
                currency=currency,
                raw=(raw or {})
            ))
        return True
    except Exception as e:
        log.exception("Checkout session insert failed: %s", e)
        return False

# --------------- reCAPTCHA helper ---------------
def verify_recaptcha(token: str, remote_ip: str | None = None):
    if not RECAPTCHA_SECRET:
        return False, {"error": "server_misconfigured: missing RECAPTCHA_SECRET"}
    try:
        r = requests.post(
            "https://www.google.com/recaptcha/api/siteverify",
            data={"secret": RECAPTCHA_SECRET, "response": token, "remoteip": remote_ip or ""},
            timeout=10,
        )
        j = r.json()
        return bool(j.get("success")), j
    except requests.RequestException as e:
        log.exception("reCAPTCHA verify error: %s", e)
        return False, {"error": "network_error"}

# --------------- Scan helpers ----------------
def genereer_zoekvragen(description: str, locatie: str, n: int = 10,
                         language: str | None = None, biz_type: str | None = None):
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
        selected = met_vraagteken if met_vraagteken else regels
        regels = selected[:n]
    return regels


def vraag_perplexity(prompt: str, return_errors: bool = False):
    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "sonar",
        "messages": [{
            "role": "user",
            "content": "Beantwoord de volgende vraag kort en concreet, in maximaal 3 zinnen. "
                       "Noem alleen bedrijven, merknamen, locaties of domeinen. Geen uitleg.\n\n" + prompt
        }],
    }
    try:
        r = requests.post("https://api.perplexity.ai/chat/completions", headers=headers, json=payload,
                          timeout=(10, PERPLEXITY_HTTP_TIMEOUT))
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


def run_vindbaarheidsscan(bedrijfsnaam: str, description: str, locatie: str, domeinnaam: str | None,
                          n: int = 10, collect: bool = False, language: str | None = None,
                          biz_type: str | None = None):
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

# --------------- API: basics ----------------
@app.route("/", methods=["GET"])
def root():
    return "scanner up", 200

@app.route("/ping", methods=["GET"])
def ping():
    return "ok", 200

# --------------- /scan ----------------
@app.route("/scan", methods=["POST", "OPTIONS"])
@cross_origin(origins=ALLOWED_ORIGINS or ["*"], methods=["POST", "OPTIONS"], allow_headers=["Content-Type"])
def scan():
    if request.method == "OPTIONS":
        return ("", 204)

    data = (request.get_json(silent=True) or request.form.to_dict() or {})
    log.info("DEBUG /scan incoming: %s", data)

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
    biz_type     = (data.get("biz_type")    or "").strip().lower() or None

    domein = (website_url.replace("https://", "").replace("http://", "").replace("/", "").lower()
              ) if website_url else None

    try:
        n = int(data.get("n", DEFAULT_MAX_N))
    except (TypeError, ValueError):
        n = DEFAULT_MAX_N
    n = max(1, min(n, DEFAULT_MAX_N))

    return_details = bool(data.get("return_details") or data.get("debug"))

    missing = []
    if not bedrijfsnaam: missing.append("company_name")
    if not description:  missing.append("description")
    if not locatie:      missing.append("location")
    if not email:        missing.append("email")
    if missing:
        return jsonify({"error": "missing required fields", "missing": missing}), 400

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
    if not ENGINE:
        return jsonify({"error": "Database niet geconfigureerd"}), 500
    try:
        with ENGINE.connect() as conn:
            result = conn.execute(text(
                "SELECT id, created_at, name, website_url, description, location, language, score, email "
                "FROM scans ORDER BY created_at DESC"
            ))
            rows = [dict(r) for r in result.mappings()]
        return jsonify(rows), 200
    except Exception as e:
        log.exception("Fout bij ophalen scans: %s", e)
        return jsonify({"error": "DB query failed"}), 500

# --------------- MAIL HELPERS ---------------
def send_mail_via_zoho(subject: str, html_body: str, reply_to: str | None = None, to_addr: str | None = None):
    if not (SMTP_USER and SMTP_PASS):
        raise RuntimeError("SMTP credentials missing")

    to_addr = to_addr or CONTACT_TO
    msg = MIMEText(html_body, "html", "utf-8")
    msg["Subject"] = subject
    msg["From"] = formataddr(("ASEO", SMTP_USER))
    msg["To"] = to_addr
    if reply_to:
        msg["Reply-To"] = reply_to

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.sendmail(SMTP_USER, [to_addr], msg.as_string())

@app.route("/contact", methods=["POST", "OPTIONS"])
@cross_origin(origins=ALLOWED_ORIGINS or ["*"], methods=["POST", "OPTIONS"], allow_headers=["Content-Type"])
def contact():
    if request.method == "OPTIONS":
        return ("", 204)
    data = (request.get_json(silent=True) or {})
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip()
    message = (data.get("message") or "").strip()
    if not (name and email and message):
        return jsonify({"error": "missing fields"}), 400
    subj = f"New contact form message from {name}"
    body = f"""
      <h3>New contact message</h3>
      <p><strong>Name:</strong> {name}<br>
         <strong>Email:</strong> {email}</p>
      <p style="white-space:pre-wrap">{message}</p>
      <hr><p>Sent from aseon.io/contact</p>
    """
    try:
        send_mail_via_zoho(subj, body, reply_to=email, to_addr=CONTACT_TO)
        return jsonify({"ok": True}), 200
    except Exception as e:
        log.exception("contact send failed: %s", e)
        return jsonify({"error": "email_failed"}), 500

# --------------- CHECKOUT START (save form + create Stripe Session) ---------------
@app.route("/checkout/start", methods=["POST", "OPTIONS"])
@cross_origin(origins=ALLOWED_ORIGINS or ["*"], methods=["POST", "OPTIONS"], allow_headers=["Content-Type"])
def checkout_start():
    if request.method == "OPTIONS":
        return ("", 204)

    if not STRIPE_SECRET:
        return jsonify({"error": "Stripe not configured"}), 500

    data = (request.get_json(silent=True) or request.form.to_dict() or {})
    log.info("DEBUG /checkout/start incoming: %s", data)

    # Fields from frontend form
    plan_key    = (data.get("plan") or "basic").strip().lower()
    full_name   = (data.get("name") or "").strip()
    company     = (data.get("company") or "").strip()
    email       = (data.get("email") or "").strip()
    competitors = (data.get("competitors") or "").strip()
    notes       = (data.get("notes") or "").strip()

    if not (full_name and email and plan_key in PLAN_CONFIG):
        return jsonify({"error": "missing or invalid fields"}), 400

    plan_cfg = PLAN_CONFIG[plan_key]
    mode = "subscription" if plan_cfg["interval"] else "payment"

    # Build line item using existing product ids + ad-hoc price_data
    price_data = {
        "currency": CURRENCY,
        "product": plan_cfg["product"],
        "unit_amount": plan_cfg["amount"],
    }
    if mode == "subscription":
        price_data["recurring"] = {"interval": plan_cfg["interval"]}

    try:
        session = stripe.checkout.Session.create(
            mode=mode,
            payment_method_types=["card"],
            line_items=[{
                "price_data": price_data,
                "quantity": 1,
            }],
            success_url=CHECKOUT_SUCCESS_URL,
            cancel_url=CHECKOUT_CANCEL_URL,
            customer_email=email,
            metadata={
                "plan": plan_cfg["name"],
                "plan_key": plan_key,
                "form_name": full_name,
                "company_name": company,
                "competitors": competitors,
                "notes": notes,
            },
            allow_promotion_codes=True,
        )

        # Save immediately for Beekeeper visibility even before webhook
        save_checkout_session(
            session_id=session["id"],
            customer_email=email,
            customer_name=full_name,
            company_name=company,
            competitors=competitors,
            notes=notes,
            plan=plan_cfg["name"],
            amount_total=plan_cfg["amount"],
            currency=CURRENCY,
            raw={"local": "created_via_checkout_start"}
        )

        return jsonify({"url": session["url"]}), 200

    except Exception as e:
        log.exception("Stripe session create failed: %s", e)
        return jsonify({"error": "stripe_error"}), 500

# --------------- STRIPE WEBHOOK ----------------
def _safe(d, *path, default=None):
    cur = d or {}
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur

def _extract_custom_fields(obj: dict) -> dict:
    out = {}
    fields = obj.get("custom_fields") or []
    for f in fields:
        key = f.get("key") or (f.get("label", {}).get("custom") or "").strip().lower().replace(" ", "_")
        val = None
        if isinstance(f.get("text"), dict):
            val = f["text"].get("value")
        elif isinstance(f.get("numeric"), dict):
            val = str(f["numeric"].get("value"))
        elif isinstance(f.get("dropdown"), dict):
            val = f["dropdown"].get("value")
        if key and (val is not None):
            out[key] = val
    return out

@app.route("/webhook/stripe", methods=["POST"])
def stripe_webhook():
    if not STRIPE_WEBHOOK_SECRET:
        log.error("STRIPE_WEBHOOK_SECRET missing")
        return ("", 500)

    payload = request.data
    sig = request.headers.get("Stripe-Signature", "")
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except stripe.error.SignatureVerificationError:
        log.warning("Stripe signature verification failed")
        return ("", 400)
    except ValueError:
        log.warning("Invalid payload")
        return ("", 400)

    etype = event.get("type", "")
    obj = event.get("data", {}).get("object", {}) or {}
    log.info("Stripe event: %s", etype)

    # Base fields
    mode = _safe(obj, "mode")
    status = _safe(obj, "status") or _safe(obj, "payment_status")
    customer_email = _safe(obj, "customer_details", "email") or _safe(obj, "customer_email")
    customer_name  = _safe(obj, "customer_details", "name")
    amount_total   = _safe(obj, "amount_total")
    currency       = _safe(obj, "currency")
    is_sub         = (mode == "subscription") or ("subscription" in etype.lower())
    plan           = (_safe(obj, "metadata", "plan") or _safe(obj, "metadata", "product"))

    # invoice.* adjustments
    if etype.startswith("invoice."):
        mode = "subscription"
        is_sub = True
        customer_email = _safe(obj, "customer_email") or customer_email
        amount_total = _safe(obj, "total") or amount_total
        currency = _safe(obj, "currency") or currency
        status = _safe(obj, "status") or status
        try:
            line = (obj.get("lines", {}).get("data") or [])[0]
            plan = _safe(line, "price", "nickname") or plan
        except Exception:
            pass

    # payment_intent fallback
    if etype == "payment_intent.succeeded":
        amount_total = _safe(obj, "amount_received") or _safe(obj, "amount")
        currency = _safe(obj, "currency") or currency
        status = "succeeded"

    # Save payment event
    save_payment_event(
        event_id=event["id"],
        event_type=etype,
        mode=mode,
        status=status,
        is_subscription=bool(is_sub),
        customer_email=customer_email,
        customer_name=customer_name,
        plan=plan,
        amount_total=amount_total,
        currency=currency,
        raw=event
    )

    # After successful checkout: persist details + send email
    if etype == "checkout.session.completed":
        try:
            session_id = obj.get("id")
            meta = obj.get("metadata") or {}
            cfields = _extract_custom_fields(obj)
            company_name = meta.get("company_name") or cfields.get("company_name") or cfields.get("company")
            competitors  = meta.get("competitors")  or cfields.get("competitors")
            notes        = meta.get("notes")        or cfields.get("notes")

            sess = stripe.checkout.Session.retrieve(session_id, expand=["line_items"])
            line_items = (sess.get("line_items") or {}).get("data", [])
            item = line_items[0] if line_items else {}
            plan_name = (item.get("description")
                         or _safe(item, "price", "nickname")
                         or plan
                         or "ASEO plan")

            cadence = "monthly subscription" if sess.get("mode") == "subscription" else "one-time purchase"

            save_checkout_session(
                session_id=session_id,
                customer_email=customer_email,
                customer_name=customer_name,
                company_name=company_name,
                competitors=competitors,
                notes=notes,
                plan=plan_name,
                amount_total=amount_total,
                currency=currency,
                raw=event
            )

            if customer_email:
                subject_user = f"Thanks — your {plan_name} ({cadence}) is confirmed"
                body_user = f"""
                  <h2>Welcome to ASEO 🎉</h2>
                  <p>Thanks for your purchase. We've received your payment for <b>{plan_name}</b> ({cadence}).</p>
                  <p>What happens next:</p>
                  <ol>
                    <li>We’ll review your details and set up your onboarding.</li>
                    <li>You’ll receive next steps by email within 1–2 business days.</li>
                  </ol>
                  <p>If you have questions, just reply to this email.</p>
                  <hr>
                  <p>— Team ASEO</p>
                """.strip()
                try:
                    send_mail_via_zoho(subject_user, body_user, to_addr=customer_email)
                except Exception as e:
                    log.exception("Failed sending user confirmation: %s", e)

        except Exception as e:
            log.exception("Error handling checkout.session.completed: %s", e)

    return jsonify({"received": True})

# --------------- Main ----------------
if __name__ == "__main__":
    # Local dev — Render gebruikt gunicorn in productie
    app.run(host="0.0.0.0", port=5000)

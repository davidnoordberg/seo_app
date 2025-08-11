import os
import time
import requests
import openai
from flask import Flask, request, jsonify

# ---- Environment (Render) ----
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PERPLEXITY_API_KEY = os.environ["PERPLEXITY_API_KEY"]
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4-turbo")  # eventueel via env instelbaar

# OpenAI client
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)

# -------- Helpers --------
def genereer_zoekvragen(description: str, locatie: str, n: int = 10):
    """Maak n natuurlijke AI-zoekvragen op basis van een vrije beschrijving."""
    try:
        n = int(n)
    except Exception:
        n = 10
    n = max(1, min(n, 12))

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
   - Eén vraag per regel.

Omschrijving:
\"\"\"{description}\"\"\"
"""

    resp = openai_client.chat.completions.create(
        model=OPENAI_MODEL,  # bv. gpt-4-turbo (default) of via env overschrijfbaar
        messages=[
            {"role": "system", "content": "Je bent een expert in gebruikerszoekgedrag en SEO."},
            {"role": "user", "content": prompt}
        ]
    )
    content = (resp.choices[0].message.content or "").strip()
    regels = [r.strip("-• ").strip() for r in content.split("\n") if r.strip()]
    if len(regels) > n:
        met_vraagteken = [r for r in regels if "?" in r]
        regels = (met_vraagteken or regels)[:n]
    return regels


def vraag_perplexity(prompt: str):
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    instructie = (
        "Beantwoord de volgende vraag kort en concreet, in maximaal 3 zinnen. "
        "Noem alleen bedrijven, merknamen, locaties of domeinen. Geen uitleg.\n\n"
    )
    payload = {
        "model": "sonar",
        "messages": [{"role": "user", "content": instructie + prompt}],
    }
    try:
        r = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload,
            timeout=45
        )
    except requests.RequestException:
        return None

    if r.status_code != 200:
        return None

    try:
        return r.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError):
        return None


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
    collect: bool = False
):
    """Als collect=True, retourneer (score, items) met Q&A."""
    vragen = genereer_zoekvragen(description, locatie, n=n)
    if not vragen:
        return 0 if not collect else (0, [])

    hits = 0
    items = []

    for vraag in vragen:
        antw = vraag_perplexity(vraag)
        hit = bool(antw and check_bedrijfsvermelding(antw, bedrijfsnaam, domeinnaam))
        if hit:
            hits += 1
        if collect:
            items.append({"q": vraag, "a": (antw or ""), "hit": hit})
        time.sleep(0.6 if n <= 3 else 1.5)  # rate limit

    score = round((hits / max(len(vragen), 1)) * 100)
    return score if not collect else (score, items)


# -------- API --------
@app.route("/ping", methods=["GET"])
def ping():
    return "ok", 200


@app.route("/scan", methods=["POST"])
def scan():
    """
    Expect JSON:
    {
      "company_name": "...",
      "description": "...",
      "location": "...",
      "website_url": "https://...",
      "n": 10,
      "return_details": true   # optioneel; als meegegeven -> ook Q&A terug
    }

    Returns normal:
      { "score": <number> }

    Returns with details (when return_details / debug is true):
      { "score": <number>, "items": [ { "q": "...", "a": "...", "hit": true } ] }
    """
    data = request.get_json(force=True) or {}

    bedrijfsnaam = (data.get("company_name") or "").strip()
    description  = (data.get("description")  or "").strip()
    locatie      = (data.get("location")    or "").strip()
    website_url  = (data.get("website_url") or "").strip()

    domein = (
        website_url.replace("https://", "").replace("http://", "").replace("/", "").lower()
    ) if website_url else None

    try:
        n = int(data.get("n", 10))
    except (TypeError, ValueError):
        n = 10
    n = max(1, min(n, 12))

    return_details = bool(data.get("return_details") or data.get("debug"))

    if not (bedrijfsnaam and description and locatie):
        return jsonify({"error": "company_name, description en location zijn verplicht"}), 400

    if return_details:
        score, items = run_vindbaarheidsscan(bedrijfsnaam, description, locatie, domein, n=n, collect=True)
        return jsonify({"score": score, "items": items}), 200
    else:
        score = run_vindbaarheidsscan(bedrijfsnaam, description, locatie, domein, n=n, collect=False)
        return jsonify({"score": score}), 200


if __name__ == "__main__":
    # Local dev — Render gebruikt gunicorn
    app.run(host="0.0.0.0", port=5000)

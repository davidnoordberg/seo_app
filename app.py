import os
import time
import requests
import openai
from flask import Flask, request, jsonify

# API keys via Render environment variables
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PERPLEXITY_API_KEY = os.environ["PERPLEXITY_API_KEY"]

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
app = Flask(__name__)

# ===== Zoekvragen generator =====
def genereer_zoekvragen(description, locatie, n=10):
    """
    Gebruikt altijd de vrije omschrijving om kernbegrippen af te leiden
    en genereert vervolgens {n} natuurlijke AI-zoekvragen.
    """
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

    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Je bent een expert in gebruikerszoekgedrag en SEO."},
            {"role": "user", "content": prompt}
        ]
    )

    content = response.choices[0].message.content or ""
    vragen = [regel.strip("-• ").strip() for regel in content.split("\n") if regel.strip()]
    if len(vragen) > n:
        kandidaten = [v for v in vragen if "?" in v]
        vragen = (kandidaten or vragen)[:n]
    return vragen


# ===== Perplexity =====
def vraag_perplexity(prompt):
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    instructie = (
        "Beantwoord de volgende vraag kort en concreet, in maximaal 3 zinnen. "
        "Noem alleen bedrijven, merknamen, locaties of domeinen. Geen uitleg.\n\n"
    )
    payload = {
        "model": "llama-3-70b-instruct",
        "messages": [{"role": "user", "content": instructie + prompt}],
    }
    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload,
            timeout=45
        )
    except requests.RequestException:
        return None

    if response.status_code != 200:
        return None

    try:
        return response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return None


def check_bedrijfsvermelding(antwoord, bedrijfsnaam, domeinnaam=None):
    if not antwoord:
        return False
    tekst = antwoord.lower()
    return bedrijfsnaam.lower() in tekst or (domeinnaam and domeinnaam.lower() in tekst)


def run_vindbaarheidsscan(bedrijfsnaam, description, locatie, domeinnaam=None, n=10):
    vragen = genereer_zoekvragen(description, locatie, n=n)
    if not vragen:
        return 0

    hits = 0
    for vraag in vragen:
        antwoord = vraag_perplexity(vraag)
        if antwoord and check_bedrijfsvermelding(antwoord, bedrijfsnaam, domeinnaam):
            hits += 1
        time.sleep(0.6 if n <= 3 else 1.5)

    return round((hits / max(len(vragen), 1)) * 100)


# ===== API =====
@app.route("/ping", methods=["GET"])
def ping():
    return "ok", 200


@app.route("/scan", methods=["POST"])
def scan_endpoint():
    """
    JSON body:
    {
      "company_name": "...",
      "description": "...",
      "location": "...",
      "website_url": "https://...",
      "n": 2
    }
    """
    data = request.get_json(force=True) or {}
    bedrijfsnaam = (data.get("company_name") or "").strip()
    description = (data.get("description") or "").strip()
    locatie = (data.get("location") or "").strip()
    website_url = (data.get("website_url") or "").strip()
    domeinnaam = (
        website_url.replace("https://", "").replace("http://", "").replace("/", "").lower()
    ) if website_url else None
    try:
        n = int(data.get("n", 10))
    except ValueError:
        n = 10

    if not (bedrijfsnaam and description and locatie):
        return jsonify({"error": "company_name, description en location zijn verplicht"}), 400

    score = run_vindbaarheidsscan(bedrijfsnaam, description, locatie, domeinnaam, n=n)
    return jsonify({"score": score})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

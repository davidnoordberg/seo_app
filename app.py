import os
import time
import requests
import openai
from flask import Flask, request, jsonify

# === API keys uit environment variables (Render → Environment) ===
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PERPLEXITY_API_KEY = os.environ["PERPLEXITY_API_KEY"]

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
app = Flask(__name__)

# ======= Originele functies, kleine aanpassing voor variabel aantal vragen =======

def genereer_zoekvragen(product, locatie, n=10):
    prompt = f"""
Je bent een SEO-expert gespecialiseerd in AI-zoekgedrag. 
Geef {n} natuurlijke vragen die iemand aan ChatGPT of Perplexity zou stellen
wanneer ze op zoek zijn naar een {product} in {locatie}.

✅ Houd de vragen:
- realistisch en veelvoorkomend
- kort en duidelijk
- gericht op zoeken en vergelijken van bedrijven
- zonder bedrijfsnamen, fictieve situaties of irrelevante details

Geef de vragen als lijst zonder uitleg of nummering.
"""
    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Je bent een expert in gebruikerszoekgedrag en SEO."},
            {"role": "user", "content": prompt}
        ]
    )

    content = response.choices[0].message.content
    vragen = [vraag.strip("-• ").strip() for vraag in content.split("\n") if vraag.strip()]
    return vragen


def vraag_perplexity(prompt):
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    instructie = (
        "Beantwoord de volgende vraag kort en concreet, in maximaal 3 zinnen. "
        "Noem alleen bedrijven, merknamen, locaties of domeinen. Geen uitleg, geen algemene adviezen.\n\n"
    )

    payload = {
        "model": "llama-3-70b-instruct",
        "messages": [{"role": "user", "content": instructie + prompt}],
    }

    response = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers=headers,
        json=payload,
        timeout=45
    )

    if response.status_code != 200:
        return None

    try:
        return response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return None


def check_bedrijfsvermelding(antwoord, bedrijfsnaam, domeinnaam=None):
    if not antwoord:
        return False
    naam_gevonden = bedrijfsnaam.lower() in antwoord.lower()
    domein_gevonden = domeinnaam.lower() in antwoord.lower() if domeinnaam else False
    return naam_gevonden or domein_gevonden


def run_vindbaarheidsscan(bedrijfsnaam, product, locatie, domeinnaam=None, n=10):
    vragen = genereer_zoekvragen(product, locatie, n=n)
    score = 0

    for vraag in vragen:
        antwoord = vraag_perplexity(vraag)
        if antwoord and check_bedrijfsvermelding(antwoord, bedrijfsnaam, domeinnaam):
            score += 1
        # Kortere sleep bij kleine n om sneller te initialiseren
        time.sleep(0.6 if n <= 3 else 1.5)

    percentage = (score / max(len(vragen), 1)) * 100
    return round(percentage)

# ======= API endpoints =======

@app.route("/ping", methods=["GET"])
def ping():
    """Handige health check om Render wakker te maken."""
    return "ok", 200

@app.route("/scan", methods=["POST"])
def scan_endpoint():
    """
    Verwacht JSON body:
    {
      "company_name": "...",
      "product_service": "...",   (of "business_category")
      "location": "...",
      "website_url": "https://...",
      "n": 3  # optioneel, aantal vragen
    }
    Retourneert: { "score": 72 }
    """
    data = request.get_json(force=True) or {}
    bedrijfsnaam = data.get("company_name", "").strip()
    product = (data.get("product_service") or data.get("business_category") or "").strip()
    locatie = data.get("location", "").strip()
    website_url = (data.get("website_url") or "").strip()
    domeinnaam = (
        website_url.replace("https://", "")
                   .replace("http://", "")
                   .replace("/", "")
                   .lower()
    ) if website_url else None

    try:
        n = int(data.get("n", 10))
    except ValueError:
        n = 10

    if not (bedrijfsnaam and product and locatie):
        return jsonify({"error": "company_name, product_service (of business_category) en location zijn verplicht"}), 400

    score = run_vindbaarheidsscan(bedrijfsnaam, product, locatie, domeinnaam, n=n)
    return jsonify({"score": score})

if __name__ == "__main__":
    # Voor lokaal testen
    app.run(host="0.0.0.0", port=5000)

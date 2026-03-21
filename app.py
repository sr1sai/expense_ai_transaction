import gradio as gr
import spacy
import pickle
import os
import zipfile
import re

# -------------------------------
# 🔧 Extract models.zip (HF Spaces)
# -------------------------------
if os.path.exists("models.zip"):
    with zipfile.ZipFile("models.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

# -------------------------------
# 📦 Load Models
# -------------------------------
nlp = spacy.load("sms_ner_model")

with open("alias_model.pkl", "rb") as f:
    alias_model = pickle.load(f)

# Compatibility shim for models pickled with newer scikit-learn versions.
if not hasattr(alias_model, "multi_class"):
    alias_model.multi_class = "auto"

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# -------------------------------
# 🧠 Alias Prediction
# -------------------------------
def predict_alias(sender, sms):
    text = sender + " " + sms
    vec = vectorizer.transform([text])

    probs = alias_model.predict_proba(vec)
    confidence = probs.max()

    if confidence < 0.6:
        return "Unknown"

    return alias_model.classes_[probs.argmax()]


def parse_amount(amount_text):
    """Parse amount text into structured format: {currency, value}"""
    if not amount_text:
        return None
    
    amount_text = str(amount_text).strip()
    
    # Extract numeric value
    numeric_match = re.search(r'\d+\.?\d*', amount_text)
    if not numeric_match:
        return None
    
    value = float(numeric_match.group())
    
    # Identify currency
    currency_map = {
        'rs': 'Rupee',
        'r': 'Rupee',
        '₹': 'Rupee',
        'inr': 'Rupee'
    }
    
    currency = 'Rupee'  # Default
    for curr_key, curr_name in currency_map.items():
        if curr_key.lower() in amount_text.lower():
            currency = curr_name
            break
    
    return {
        "currency": currency,
        "value": value
    }

# -------------------------------
# 🧾 Main Parser
# -------------------------------
def parse_sms(sender, sms):
    doc = nlp(sms)

    result = {
        "Type": None,
        "Amount": None,
        "Target": None,
        "Account": None,
        "Alias": None
    }

    bank = None
    account = None

    for ent in doc.ents:
        if ent.label_ == "TYPE":
            result["Type"] = ent.text
        elif ent.label_ == "AMOUNT":
            result["Amount"] = parse_amount(ent.text)
        elif ent.label_ == "TARGET":
            result["Target"] = ent.text
        elif ent.label_ == "ACCOUNT":
            account = ent.text
        elif ent.label_ == "BANK":
            bank = ent.text.upper()

    if bank and account:
        result["Account"] = f"{bank}:{account}"
    else:
        result["Account"] = account

    result["Alias"] = predict_alias(sender, sms)

    return result

# -------------------------------
# 🌐 Gradio UI + API
# -------------------------------
demo = gr.Interface(
    fn=parse_sms,
    inputs=[
        gr.Textbox(label="Sender"),
        gr.Textbox(label="SMS Content")
    ],
    outputs=gr.JSON(label="Parsed Output"),
    title="📩 Bank SMS Parser (NER + Alias ML)"
)

# -------------------------------
# 🚀 Launch
# -------------------------------
demo.launch()
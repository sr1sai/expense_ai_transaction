import gradio as gr
import spacy
import pickle
import os
import zipfile

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
            result["Amount"] = ent.text
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
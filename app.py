import joblib
import gradio as gr
import re

# ------------------------
# Load models
# ------------------------

type_model = joblib.load("type_model.joblib")
target_model = joblib.load("target_model.joblib")
alias_model = joblib.load("alias_model.joblib")
account_model = joblib.load("account_model.joblib")


# ------------------------
# Clean text
# ------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r'\d+', ' number ', text)
    text = re.sub(r'[₹$]', ' currency ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ------------------------
# Extract amount
# ------------------------

def extract_amount(message):

    match = re.search(r'(?:rs\.?|₹)\s?([\d,]+(?:\.\d+)?)', message.lower())

    if match:
        return float(match.group(1).replace(",", ""))

    return None


# ------------------------
# Prediction
# ------------------------

def predict(sender, message):

    text = clean_text(sender + " " + message)

    result = {
        "Type": type_model.predict([text])[0],
        "Amount": extract_amount(message),
        "Target": target_model.predict([text])[0],
        "Alias": alias_model.predict([text])[0],
        "Account": account_model.predict([text])[0]
    }

    return result


# ------------------------
# Gradio interface
# ------------------------

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Sender"),
        gr.Textbox(label="Message Content", lines=4)
    ],
    outputs="json",
    title="SMS Transaction Extractor"
)

demo.launch()
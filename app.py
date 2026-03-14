import gradio as gr
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

# --------------------
# Load model
# --------------------
MODEL_NAME = "srisaidivyakola/transaction_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Model loaded successfully")

# --------------------
# Load dataset & stats
# --------------------
df = pd.read_csv("Lables.csv")

total_records = len(df)
debit_count = (df["Type"] == "Debit").sum()
credit_count = (df["Type"] == "Credit").sum()

stats_text = (
    f"{{total records: {total_records} "
    f"Debit: {debit_count} "
    f"Credit: {credit_count}}}"
)

print(stats_text)

# --------------------
# Prediction function
# --------------------
def predict(sender, message):

    text = f"extract transaction details: Sender: {sender} Message: {message}"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(**inputs, max_length=128)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        parsed = json.loads(result)
        return parsed
    except:
        return {"error": result}

# --------------------
# Gradio UI
# --------------------
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Sender", placeholder="AX-ICICIT-S"),
        gr.Textbox(label="Message Content", lines=4)
    ],
    outputs="json",
    title="SMS Transaction Extractor",
    description=(
        "Extracts transaction details from SMS\n\n"
        f"📊 Dataset Stats:\n{stats_text}"
    )
)

demo.launch()
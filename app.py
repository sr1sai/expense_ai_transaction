import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "srisaidivyakola/transaction_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def predict(message, sender):

    text = f"extract transaction details: Sender: {sender} Message: {message}"

    inputs = tokenizer(text, return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(**inputs, max_length=128)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return result


def api_predict(message, sender):
    return predict(message, sender)


demo = gr.Interface(
    fn=api_predict,
    inputs=[
        gr.Textbox(label="Sender"),
        gr.Textbox(label="Message")
    ],
    outputs=gr.Textbox(label="Extracted Transaction"),
    title="SMS Transaction Extractor",
    description="Extract Type, Amount, Target, Alias and Account from SMS"
)

demo.launch(server_name="0.0.0.0", server_port=7860)
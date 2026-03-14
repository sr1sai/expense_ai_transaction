from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


APP_DIR = Path(__file__).resolve().parent
LOCAL_MODEL_DIR = APP_DIR / "transaction_model_export"
PROMPT_TEMPLATE = "extract transaction details: Sender: {sender} Message: {message}"
MAX_INPUT_LENGTH = 256
MAX_OUTPUT_LENGTH = 128
OUTPUT_KEYS = ("type", "amount", "target", "alias", "account")


def resolve_model_source() -> str:
    if LOCAL_MODEL_DIR.exists():
        return str(LOCAL_MODEL_DIR)

    model_id = os.getenv("MODEL_ID", "").strip()
    if model_id:
        return model_id

    raise FileNotFoundError(
        "No model found. Add a transaction_model_export folder to this Space or set the MODEL_ID "
        "environment variable to your fine-tuned Hugging Face model repo."
    )


@lru_cache(maxsize=1)
def load_bundle() -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM, str, str]:
    model_source = resolve_model_source()
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_source)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device, model_source


def parse_prediction(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "type": None,
            "amount": None,
            "target": None,
            "alias": None,
            "account": None,
            "raw_output": cleaned,
        }

    normalized: dict[str, Any] = {key: parsed.get(key) for key in OUTPUT_KEYS}
    normalized["raw_output"] = cleaned
    return normalized


def predict_transaction(message: str, sender: str) -> tuple[dict[str, Any], str, str]:
    if not message or not message.strip():
        raise gr.Error("Message is required.")

    safe_sender = sender.strip() if sender and sender.strip() else "UNKNOWN"

    try:
        tokenizer, model, device, model_source = load_bundle()
    except FileNotFoundError as exc:
        raise gr.Error(str(exc)) from exc

    prompt = PROMPT_TEMPLATE.format(sender=safe_sender, message=message.strip())
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_length=MAX_OUTPUT_LENGTH)

    raw_prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    parsed_prediction = parse_prediction(raw_prediction)
    return parsed_prediction, json.dumps(parsed_prediction, indent=2), model_source


with gr.Blocks(title="Expense AI Transaction Extractor") as demo:
    gr.Markdown(
        """
        # Expense AI Transaction Extractor
        Fine-tuned FLAN-T5 inference for bank SMS transaction extraction.

        This Space exposes a Gradio API named `predict_transaction`, so you can use the UI or call it as an endpoint.
        """
    )

    with gr.Row():
        sender_input = gr.Textbox(label="Sender", placeholder="AX-ICICIT-S")
        message_input = gr.Textbox(
            label="Message",
            lines=6,
            placeholder="Paste the bank SMS text here.",
        )

    run_button = gr.Button("Extract transaction")

    structured_output = gr.JSON(label="Structured output")
    raw_output = gr.Textbox(label="JSON payload", lines=8)
    model_output = gr.Textbox(label="Loaded model", interactive=False)

    gr.Examples(
        examples=[
            [
                "AX-ICICIT-S",
                "ICICI Bank Acct XX789 debited for Rs 80.00 on 26-Dec-25; AWFIS credited. UPI:590283159319.",
            ],
            [
                "JD-UNIONB-S",
                "A/c *0186 Credited for Rs:5000.00 on 26-12-2025 21:11:28 by Mob Bk ref no 696279687099 Avl Bal Rs:5818.22.",
            ],
            [
                "VM-ICICIT-S",
                "Rs. 90.00 debited from ICICI Bank Acc XX789 on 01-Jan-26 MIN*DIGITALOC. Bal Rs. 2,701.41.",
            ],
        ],
        inputs=[sender_input, message_input],
    )

    run_button.click(
        fn=predict_transaction,
        inputs=[message_input, sender_input],
        outputs=[structured_output, raw_output, model_output],
        api_name="predict_transaction",
    )

    gr.Markdown(
        """
        Local model loading order:
        1. `transaction_model_export/` in this repo
        2. `MODEL_ID` Space variable pointing to a Hugging Face model repo
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
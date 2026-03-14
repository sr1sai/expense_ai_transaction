---
title: Expense AI Transaction Extractor
emoji: "💸"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.20.1
app_file: app.py
pinned: false
---

# Expense AI Transaction Extractor

This Space turns the notebook-trained SMS extractor into a deployable Gradio app with an API endpoint.

## What this repo needs

The Space app expects one of these model sources:

1. A local `transaction_model_export/` folder created from the notebook after training.
2. A `MODEL_ID` Space variable pointing to your fine-tuned Hugging Face model repo.

## Export the model from the notebook

After training completes in the notebook, save the final model and tokenizer with:

```python
MODEL_EXPORT_DIR = "transaction_model_export"
trainer.save_model(MODEL_EXPORT_DIR)
tokenizer.save_pretrained(MODEL_EXPORT_DIR)
```

If you want the Space to work without a separate model repo, add that exported folder to this repository before deployment.

## Endpoint usage

The app exposes a Gradio API named `predict_transaction`.

Python client example after deployment:

```python
from gradio_client import Client

client = Client("your-username/your-space-name")
result = client.predict(
    message="ICICI Bank Acct XX789 debited for Rs 80.00 on 26-Dec-25; AWFIS credited.",
    sender="AX-ICICIT-S",
    api_name="/predict_transaction",
)
print(result)
```

## Files added for the Space

- `app.py`: Gradio UI plus API endpoint.
- `requirements.txt`: Runtime dependencies for Hugging Face Spaces.
- `Extractor.ipynb`: your training notebook.

## Notes

- The model prompt format in the app matches the notebook training prompt.
- If the model does not emit valid JSON, the app still returns the raw output so you can debug prompt quality or model behavior.
---
title: SMS Transaction Extractor
emoji: 💳
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.36.1"
python_version: "3.10"
app_file: app.py
pinned: false
---

# SMS Transaction Extractor

This model extracts transaction details from SMS messages.

## Input

Sender  
Example: AX-ICICIT-S

Message  
Example:  
ICICI Bank Acct XX789 debited for Rs 80.00 on 26-Dec-25; AWFIS credited.

## Output

```json
{
"type":"Debit",
"amount":80,
"target":"AWFIS",
"alias":"Coworking",
"account":"ICICI"
}
# SMS Transaction Extractor

This model extracts structured transaction information from SMS.

Fields extracted:

- Type
- Amount
- Target
- Alias
- Account

Example Input:

Sender:
AX-ICICIT-S

Message:
ICICI Bank Acct XX789 debited for Rs 80.00 on 26-Dec-25; AWFIS credited

Example Output:

{
"type":"Debit",
"amount":80,
"target":"AWFIS",
"alias":"Coworking",
"account":"ICICI"
}
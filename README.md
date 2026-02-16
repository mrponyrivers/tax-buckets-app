# Tax Buckets — Statement Parser + Categorizer (Streamlit)

Upload or paste bank/credit-card statements → auto-categorize with rules → review/edit → export CSV/XLSX for accountant / QuickBooks workflows.

## Features
- Paste Chase statement text OR upload CSV/XLSX/PDF
- Rule-based auto-categorization (editable rules table)
- Review/edit buckets/categories + notes
- Exports:
  - Full categorized export (CSV/XLSX)
  - QuickBooks-style bank feed export (CSV/XLSX)
- Reports: spend by category, top merchants, FX fees, assistant pay totals

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py

```

```

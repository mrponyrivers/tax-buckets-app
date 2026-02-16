# Tax Buckets — Statement Parser + Categorizer (Streamlit)

Upload or paste bank/credit-card statements → auto-categorize with rules → review/edit → export CSV/XLSX for accountant / QuickBooks workflows.

Live demo: https://mrponyrivers-tax-buckets-app.streamlit.app/

## Try it quickly (sample statement)
1) Open the app
2) Go to **Import / Paste**
3) Paste the contents of:
   - `sample_data/sample_statement.txt`
4) Click **Parse pasted statement**
5) Go to **Review + Export** and test edits + downloads

## Features
- Paste Chase statement text OR upload CSV/XLSX/PDF
- Rule-based auto-categorization (editable rules table)
- Review/edit buckets/categories + notes
- Exports:
  - Full categorized export (CSV/XLSX)
  - QuickBooks-style bank feed export (CSV/XLSX)
- Reports: spend by category, top merchants, FX fees, assistant pay totals
## Screenshots
![App screenshot](docs/screenshot.png)

## Run locally

pip install -r requirements.txt
streamlit run app.py

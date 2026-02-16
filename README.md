# Tax Buckets — Statement Parser + Categorizer (Streamlit)

Upload or paste bank/credit-card statements → auto-categorize with rules → review/edit → export CSV/XLSX for accountant / QuickBooks workflows.

**Live demo:** https://mrponyrivers-tax-buckets-app.streamlit.app/

---

## What this app does
This Streamlit app helps turn messy statement text (or CSV/XLSX/PDF exports) into a clean, reviewable transaction table:
- Extracts transactions (best-effort parsing for Chase statement text / PDFs)
- Auto-labels **Bucket / Category / Subcategory** using editable rules
- Supports review/editing with manual overrides
- Exports files ready for accountant workflows and QuickBooks bank feeds

---

## Try it quickly (sample statement)
1) Open the live demo
2) Go to **Import / Paste**
3) Open `sample_data/sample_statement.txt` in GitHub and copy its contents
4) Paste into the text box
5) Click **Parse pasted statement**
6) Go to **Review + Export** and test edits + downloads

Tip: Try adding a new rule in the **Rules** tab (ex: keyword “uber”) and re-apply rules to see categorization update.

---

## Features
- Paste Chase statement text OR upload CSV/XLSX/PDF
- Rule-based auto-categorization (editable rules table)
- Review/edit buckets/categories + notes
- Manual override field for categories
- Exports:
  - Full categorized export (CSV/XLSX)
  - QuickBooks-style bank feed export (CSV/XLSX)
- Reports: spend by category, top merchants, FX fees, assistant pay totals

---
## Tech stack
- Python, Streamlit
- pandas (data wrangling)
- pdfplumber (PDF text extraction)
- openpyxl (Excel export)

## How it works
1) Import statement text (paste) or upload CSV/XLSX/PDF  
2) Parser extracts transactions into a normalized table  
3) Rules engine assigns Bucket/Category/Subcategory + confidence  
4) Review screen lets you edit + override categories  
5) Export either:
   - Full categorized ledger (CSV/XLSX)
   - QuickBooks-style bank feed (CSV/XLSX)

## Privacy
This app runs locally or on Streamlit Cloud and does **not** require connecting to any bank accounts.
Use redacted statements for demos and avoid uploading sensitive personal info to public deployments.

## Roadmap
- Better PDF parsing for more statement formats
- Rule suggestions from review edits
- Per-client “projects” presets
- Optional: chart exports (monthly spend, category trends)

## Screenshots
![App screenshot](docs/screenshot.png)

---

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py

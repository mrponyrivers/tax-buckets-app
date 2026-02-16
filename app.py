import re
import json
from io import BytesIO
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import pandas as pd
import streamlit as st
import pdfplumber


# ============================================================
# Core helpers
# ============================================================
def normalize_ws(s: str) -> str:
    return " ".join((s or "").split()).strip()


def safe_float(x) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    s = s.replace("$", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return None


def parse_mmdd_to_date(mmdd: str, year: int) -> str:
    mmdd = mmdd.strip()
    return f"{mmdd}/{year}"


def is_outgoing_transfer(desc: str) -> bool:
    """Best-effort: identify outgoing payment app transfers."""
    d = (desc or "").lower()
    if "payment to" in d:
        return True
    if "payment sent" in d:
        return True
    if "zelle payment to" in d:
        return True
    if "venmo" in d and "visa direct" in d:
        return True
    if "iat paypal" in d or "paypalsec" in d:
        return True
    if "wire transfer" in d:
        return True
    return False


def is_incoming_transfer(desc: str) -> bool:
    d = (desc or "").lower()
    if "payment from" in d:
        return True
    if "zelle payment from" in d:
        return True
    if "remote online deposit" in d:
        return True
    if "book transfer credit" in d or "foreign remittance credit" in d:
        return True
    return False


def ensure_cols(df: pd.DataFrame, cols: List[str], default="") -> pd.DataFrame:
    """Ensure df has the listed columns."""
    if df is None:
        return df
    for c in cols:
        if c not in df.columns:
            df[c] = default
    return df


# ============================================================
# A) Dedupe
# ============================================================
def dedupe_transactions(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    d = df.copy()

    for c in ["Account", "Section", "Transaction date", "Original Description"]:
        if c not in d.columns:
            d[c] = ""

    if "Amount" not in d.columns:
        d["Amount"] = None

    d["_dedupe_key"] = (
        d["Account"].astype(str).str.strip().str.lower()
        + "|"
        + d["Section"].astype(str).str.strip().str.lower()
        + "|"
        + d["Transaction date"].astype(str).str.strip()
        + "|"
        + d["Amount"].astype(str).str.strip()
        + "|"
        + d["Original Description"].astype(str).str.strip().str.lower()
    )

    d = d.drop_duplicates(subset=["_dedupe_key"], keep="first").drop(columns=["_dedupe_key"])
    return d


# ============================================================
# B) Merchant extraction
# ============================================================
def merchant_from_description(desc: str) -> str:
    d = normalize_ws(desc)
    if not d:
        return ""

    # ACH / Electronic: Orig CO Name:
    m = re.search(r"orig co name\s*:\s*(.+?)\s+orig id\s*:", d, flags=re.I)
    if m:
        return normalize_ws(m.group(1)).upper()

    low = d.lower()

    if "uber" in low:
        return "UBER"
    if "lyft" in low:
        return "LYFT"
    if "doordash" in low or "dd *doordash" in low:
        return "DOORDASH"
    if "deliveroo" in low:
        return "DELIVEROO"
    if "spotify" in low:
        return "SPOTIFY"
    if "netflix" in low:
        return "NETFLIX"
    if "amazon" in low:
        return "AMAZON"
    if "venmo" in low:
        return "VENMO"
    if "paypal" in low or "paypalsec" in low or "iat paypal" in low:
        return "PAYPAL"
    if "zelle" in low:
        return "ZELLE"
    if "wire transfer" in low:
        return "WIRE TRANSFER"
    if "applecard gsbank" in low:
        return "APPLE CARD"
    if "metergy solution" in low:
        return "METERGY SOLUTION"

    cleaned = re.sub(
        r"\b(card|purchase|recurring|payment|sent|with|pin|online|pending)\b",
        " ",
        d,
        flags=re.I,
    )
    cleaned = normalize_ws(cleaned)

    cleaned = re.sub(r"\bCard\s+\d{3,6}\b.*$", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"\b\d{2}/\d{2}\b", "", cleaned).strip()
    cleaned = cleaned.replace("*", " ")
    cleaned = normalize_ws(cleaned)

    parts = cleaned.split()
    if not parts:
        return ""
    return " ".join(parts[:5]).upper()


# ============================================================
# Categories + Rules
# ============================================================
DEFAULT_BUCKETS = [
    "Business",
    "Personal",
    "Uncategorized / Needs Review",
    "Split Needed",
    "Transfer (Internal)",
]

DEFAULT_CATEGORIES = [
    # Business
    "Travel (air, hotel, taxis)",
    "Meals (client meals)",
    "Supplies (hair supplies, wigs, extensions)",
    "Assistant Pay",
    "Business Rent",
    "Storage",
    "Insurance",
    "Subscriptions",
    "Office supplies",
    "Fees (bank/FX/transfer)",
    "Business Income",
    "Loan",
    "Loan: SBA EIDL",
    # Personal
    "Groceries",
    "Dining",
    "Health",
    "Beauty/cosmetics",
    "Pets",
    "Entertainment",
    "Personal Rent/mortgage",
    "Needs Review",
]


def default_rules() -> List[Dict]:
    return [
        # ===== confirmed mappings =====
        {"keyword": "sba eidl", "bucket": "Business", "category": "Loan: SBA EIDL", "subcategory": "Loan Payment", "confidence": 0.99},
        {"keyword": "eidl loan", "bucket": "Business", "category": "Loan: SBA EIDL", "subcategory": "Loan Payment", "confidence": 0.99},

        {"keyword": "applecard gsbank", "bucket": "Business", "category": "Supplies (hair supplies, wigs, extensions)", "subcategory": "Apple Card", "confidence": 0.95},
        {"keyword": "metergy solution", "bucket": "Business", "category": "Business Rent", "subcategory": "Utilities/Rent", "confidence": 0.95},

        # ===== major patterns =====
        {"keyword": "uber", "bucket": "Business", "category": "Travel (air, hotel, taxis)", "subcategory": "Uber/Lyft", "confidence": 0.90},
        {"keyword": "lyft", "bucket": "Business", "category": "Travel (air, hotel, taxis)", "subcategory": "Uber/Lyft", "confidence": 0.90},

        {"keyword": "doordash", "bucket": "Business", "category": "Meals (client meals)", "subcategory": "Delivery", "confidence": 0.80},
        {"keyword": "deliveroo", "bucket": "Business", "category": "Meals (client meals)", "subcategory": "Delivery", "confidence": 0.80},

        {"keyword": "spotify", "bucket": "Business", "category": "Subscriptions", "subcategory": "Spotify", "confidence": 0.70},
        {"keyword": "netflix", "bucket": "Business", "category": "Subscriptions", "subcategory": "Netflix", "confidence": 0.70},
        {"keyword": "nytimes", "bucket": "Business", "category": "Subscriptions", "subcategory": "NYTimes", "confidence": 0.70},

        {"keyword": "foreign exch rt adj fee", "bucket": "Business", "category": "Fees (bank/FX/transfer)", "subcategory": "FX Fee", "confidence": 0.98},

        {"keyword": "amazon", "bucket": "Business", "category": "Supplies (hair supplies, wigs, extensions)", "subcategory": "Amazon", "confidence": 0.70},

        {"keyword": "t-mobile", "bucket": "Business", "category": "Office supplies", "subcategory": "Phone/Internet", "confidence": 0.80},
        {"keyword": "usps", "bucket": "Business", "category": "Office supplies", "subcategory": "Postage/Shipping", "confidence": 0.80},

        {"keyword": "delta air", "bucket": "Business", "category": "Travel (air, hotel, taxis)", "subcategory": "Flights", "confidence": 0.80},
        {"keyword": "jfk international", "bucket": "Business", "category": "Travel (air, hotel, taxis)", "subcategory": "Airport", "confidence": 0.70},

        {"keyword": "prime storage", "bucket": "Business", "category": "Storage", "subcategory": "Storage", "confidence": 0.75},
        {"keyword": "shurgard", "bucket": "Business", "category": "Storage", "subcategory": "Storage", "confidence": 0.75},
    ]


def apply_rules(df: pd.DataFrame, rules: List[Dict]) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    d = df.copy()
    for col, default in [("Bucket", ""), ("Category", ""), ("Subcategory", ""), ("Confidence", 0.0)]:
        if col not in d.columns:
            d[col] = default

    # 1) keyword rules
    for idx, row in d.iterrows():
        desc = normalize_ws(row.get("Original Description", ""))
        low = desc.lower()

        matched = False
        for r in rules:
            kw = (r.get("keyword") or "").lower()
            if kw and kw in low:
                d.at[idx, "Bucket"] = r["bucket"]
                d.at[idx, "Category"] = r["category"]
                d.at[idx, "Subcategory"] = r.get("subcategory", "")
                d.at[idx, "Confidence"] = float(r.get("confidence", 0.0))
                matched = True
                break

        if not matched:
            d.at[idx, "Bucket"] = d.at[idx, "Bucket"] or ""
            d.at[idx, "Category"] = d.at[idx, "Category"] or ""
            d.at[idx, "Subcategory"] = d.at[idx, "Subcategory"] or ""
            d.at[idx, "Confidence"] = float(d.at[idx, "Confidence"] or 0.0)

    # 2) transfers logic (payment apps)
    for idx, row in d.iterrows():
        desc = (row.get("Original Description", "") or "")
        low = desc.lower()

        payment_app = any(k in low for k in ["venmo", "paypal", "zelle"])
        outgoing = is_outgoing_transfer(desc)
        incoming = is_incoming_transfer(desc)

        # Outgoing payment apps => Assistant Pay (default)
        if payment_app and outgoing:
            if not d.at[idx, "Bucket"] or d.at[idx, "Bucket"] == "Uncategorized / Needs Review":
                d.at[idx, "Bucket"] = "Business"
                d.at[idx, "Category"] = "Assistant Pay"
                d.at[idx, "Subcategory"] = "Payment App Outgoing"
                d.at[idx, "Confidence"] = max(float(d.at[idx, "Confidence"] or 0.0), 0.75)
            continue

        # Incoming payment apps => Business Income (default)
        if payment_app and incoming:
            if not d.at[idx, "Bucket"] or d.at[idx, "Bucket"] == "Uncategorized / Needs Review":
                d.at[idx, "Bucket"] = "Business"
                d.at[idx, "Category"] = "Business Income"
                d.at[idx, "Subcategory"] = "Payment App Incoming"
                d.at[idx, "Confidence"] = max(float(d.at[idx, "Confidence"] or 0.0), 0.65)
            continue

        # wire transfers -> Fees by default (unless already matched)
        if "wire transfer" in low and (not d.at[idx, "Bucket"] or d.at[idx, "Bucket"] == "Uncategorized / Needs Review"):
            d.at[idx, "Bucket"] = "Business"
            d.at[idx, "Category"] = "Fees (bank/FX/transfer)"
            d.at[idx, "Subcategory"] = "Wire"
            d.at[idx, "Confidence"] = max(float(d.at[idx, "Confidence"] or 0.0), 0.85)
            continue

        if not d.at[idx, "Bucket"]:
            d.at[idx, "Bucket"] = "Uncategorized / Needs Review"
            d.at[idx, "Category"] = "Needs Review"
            d.at[idx, "Subcategory"] = ""
            d.at[idx, "Confidence"] = 0.0

    return d


def apply_manual_override(df: pd.DataFrame, manual_col: str, target_col: str) -> pd.DataFrame:
    """If manual_col has text, override target_col with it."""
    if df is None or df.empty:
        return df
    if manual_col not in df.columns:
        return df
    df[manual_col] = df[manual_col].fillna("").astype(str).str.strip()
    mask = df[manual_col] != ""
    if target_col in df.columns:
        df.loc[mask, target_col] = df.loc[mask, manual_col]
    else:
        df[target_col] = ""
        df.loc[mask, target_col] = df.loc[mask, manual_col]
    return df


def enrich_transactions(df: pd.DataFrame, rules: List[Dict]) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    d = df.copy()
    d = dedupe_transactions(d)

    d["Merchant"] = d["Original Description"].apply(merchant_from_description)
    d["Has FX"] = d["Original Description"].str.contains(r"\b(euro|pound|exchg|foreign)\b", case=False, na=False)

    for col in ["Project", "Assistant", "Notes", "Category_Manual"]:
        if col not in d.columns:
            d[col] = ""

    d = apply_rules(d, rules)
    d = apply_manual_override(d, manual_col="Category_Manual", target_col="Category")
    d = dedupe_transactions(d)
    return d


# ============================================================
# Parsing Chase statement from text
# ============================================================
SECTION_HEADERS = [
    ("DEPOSITS AND ADDITIONS", "Deposits and Additions"),
    ("ATM & DEBIT CARD WITHDRAWALS", "Card Purchases"),
    ("ELECTRONIC WITHDRAWALS", "Electronic Withdrawals"),
    ("FEES", "Fees"),
]


def detect_section(line: str) -> Optional[str]:
    u = line.upper()
    for key, label in SECTION_HEADERS:
        if key in u:
            return label
    return None


def parse_chase_statement_text(statement_text: str, account_label: str, year: int) -> pd.DataFrame:
    raw_lines = [normalize_ws(x) for x in (statement_text or "").splitlines()]
    lines = [x for x in raw_lines if x]

    rows = []
    current_section = ""
    buffer_desc = ""
    buffer_date = ""
    buffer_amount = None

    amt_re = re.compile(r"(?P<amt>\$?\d[\d,]*\.\d{2})\s*$")
    date_re = re.compile(r"^(?P<mmdd>\d{2}/\d{2})\b")

    def flush_buffer():
        nonlocal buffer_desc, buffer_date, buffer_amount, current_section
        if buffer_date and buffer_amount is not None and buffer_desc:
            rows.append(
                {
                    "Account": account_label,
                    "Section": current_section or "Statement",
                    "Transaction date": parse_mmdd_to_date(buffer_date, year),
                    "Original Description": buffer_desc,
                    "Amount": float(buffer_amount),
                }
            )
        buffer_desc = ""
        buffer_date = ""
        buffer_amount = None

    for line in lines:
        sec = detect_section(line)
        if sec:
            flush_buffer()
            current_section = sec
            continue

        if line.lower().startswith("total "):
            continue

        dm = date_re.match(line)
        if dm:
            flush_buffer()
            buffer_date = dm.group("mmdd")

            am = amt_re.search(line)
            if am:
                buffer_amount = safe_float(am.group("amt"))
                desc = re.sub(r"^\d{2}/\d{2}\s*", "", line)
                desc = amt_re.sub("", desc).strip()
                buffer_desc = desc
            else:
                buffer_desc = re.sub(r"^\d{2}/\d{2}\s*", "", line).strip()
                buffer_amount = None
            continue

        if buffer_date:
            am = amt_re.search(line)
            if am and buffer_amount is None:
                buffer_amount = safe_float(am.group("amt"))
                extra = amt_re.sub("", line).strip()
                if extra:
                    buffer_desc = normalize_ws(buffer_desc + " " + extra)
            else:
                buffer_desc = normalize_ws(buffer_desc + " " + line)

    flush_buffer()

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["Account", "Section", "Transaction date", "Original Description", "Amount"])
    return dedupe_transactions(df)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    text_pages = []
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                text_pages.append(txt)
    return "\n".join(text_pages)


# ============================================================
# Exports
# ============================================================
def to_quickbooks_bank_feed(df: pd.DataFrame, expenses_negative: bool = True) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Date", "Description", "Amount"])

    d = df.copy()

    def qb_amount(row) -> float:
        amt = float(row.get("Amount", 0.0) or 0.0)
        section = str(row.get("Section", "")).lower()

        if not expenses_negative:
            return amt

        if "card purchases" in section or "electronic withdrawals" in section or "fees" in section:
            return -abs(amt)
        if "deposits" in section:
            return abs(amt)
        return amt

    out = pd.DataFrame()
    out["Date"] = d["Transaction date"]
    out["Description"] = d["Original Description"]
    out["Amount"] = d.apply(qb_amount, axis=1)
    return out


def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Export") -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return out.getvalue()


# ============================================================
# Session persistence (Save/Load Review)
# ============================================================
def export_session_payload() -> Dict:
    payload = {
        "rules": st.session_state.get("rules", default_rules()),
        "df_raw": None,
        "df_enriched": None,
        "df_review": None,
        "meta": {
            "saved_at": datetime.now().isoformat(),
            "app_version": "v5-manual-category-override-loans-fixed-save",
        },
    }
    for k in ["df_raw", "df_enriched", "df_review"]:
        if k in st.session_state and isinstance(st.session_state[k], pd.DataFrame):
            payload[k] = st.session_state[k].to_dict(orient="records")
    return payload


def import_session_payload(payload: Dict) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    rules = payload.get("rules")
    if isinstance(rules, list) and rules:
        st.session_state["rules"] = rules

    def to_df(key: str) -> Optional[pd.DataFrame]:
        data = payload.get(key)
        if isinstance(data, list):
            return pd.DataFrame(data)
        return None

    df_raw = to_df("df_raw")
    df_enriched = to_df("df_enriched")
    df_review = to_df("df_review")
    return df_raw, df_enriched, df_review


# ============================================================
# Streamlit UI (polished, no regression)
# ============================================================
st.set_page_config(page_title="Tax Buckets — Statement Parser + Categorizer", layout="wide")

st.title("Tax Buckets — QuickBooks-style Statement Parser + Categorizer")
st.write("**Goal:** paste/upload statements → auto-categorize → review → export CSV for accountant/QB workflows.")

# init rules
if "rules" not in st.session_state:
    st.session_state["rules"] = default_rules()

# ---------- Top Nav ----------
nav = st.tabs(["Import / Paste", "Rules", "Review + Export", "Reports"])


# ============================================================
# TAB 1: Import / Paste
# ============================================================
with nav[0]:
    st.subheader("Import options")

    colL, colR = st.columns([1, 1], gap="large")

    # A) Paste
    with colL:
        st.markdown("### A) Paste Chase statement text")
        account_label = st.text_input("Account label", value="Chase Business (9913)", key="acct_label_paste")
        year = st.number_input("Year", min_value=2000, max_value=2100, value=2025, step=1, key="year_paste")

        statement_text = st.text_area("Paste statement text here", height=260)

        if st.button("Parse pasted statement", key="parse_paste_btn"):
            df_raw = parse_chase_statement_text(statement_text, account_label=account_label, year=int(year))
            st.session_state["df_raw"] = df_raw
            st.session_state.pop("df_enriched", None)
            st.session_state.pop("df_review", None)
            st.success(f"Parsed {len(df_raw)} rows from pasted text.")

    # B) Upload
    with colR:
        st.markdown("### B) Upload file (CSV / XLSX / PDF)")
        st.caption("PDF parsing is best-effort: we extract text and run the same parser.")

        account_label_up = st.text_input("Account label for upload", value="Chase Business (9913)", key="acct_label_upload")
        year_up = st.number_input("Year for upload", min_value=2000, max_value=2100, value=2025, step=1, key="year_upload")
        uploaded = st.file_uploader("Upload a statement file", type=["csv", "xlsx", "pdf"], key="uploader")

        if uploaded is not None:
            filename = uploaded.name.lower()
            df_raw = None

            if filename.endswith(".csv"):
                df_raw = pd.read_csv(uploaded)
            elif filename.endswith(".xlsx"):
                df_raw = pd.read_excel(uploaded)
            elif filename.endswith(".pdf"):
                pdf_bytes = uploaded.read()
                text = extract_text_from_pdf(pdf_bytes)
                df_raw = parse_chase_statement_text(text, account_label=account_label_up, year=int(year_up))
            else:
                st.error("Unsupported file type.")

            if df_raw is not None:
                # normalize columns if file is a transaction export
                if "Original Description" not in df_raw.columns:
                    cols = {c.lower(): c for c in df_raw.columns}
                    if "memo/description" in cols:
                        df_raw["Original Description"] = df_raw[cols["memo/description"]].astype(str)
                    elif "description" in cols:
                        df_raw["Original Description"] = df_raw[cols["description"]].astype(str)

                if "Transaction date" not in df_raw.columns:
                    cols = {c.lower(): c for c in df_raw.columns}
                    if "transaction date" in cols:
                        df_raw["Transaction date"] = df_raw[cols["transaction date"]].astype(str)
                    elif "date" in cols:
                        df_raw["Transaction date"] = df_raw[cols["date"]].astype(str)

                if "Amount" not in df_raw.columns:
                    cols = {c.lower(): c for c in df_raw.columns}
                    if "amount" in cols:
                        df_raw["Amount"] = df_raw[cols["amount"]].apply(safe_float)

                if "Account" not in df_raw.columns:
                    df_raw["Account"] = account_label_up
                if "Section" not in df_raw.columns:
                    df_raw["Section"] = "Statement"

                df_raw = dedupe_transactions(df_raw)
                st.session_state["df_raw"] = df_raw
                st.session_state.pop("df_enriched", None)
                st.session_state.pop("df_review", None)
                st.success(f"Parsed {len(df_raw)} rows from upload.")

    st.divider()

    if "df_raw" in st.session_state and isinstance(st.session_state["df_raw"], pd.DataFrame) and not st.session_state["df_raw"].empty:
        st.markdown("### Parsed preview")
        st.caption(f"Rows: {len(st.session_state['df_raw'])}")
        st.dataframe(st.session_state["df_raw"].head(200), width='stretch')
    else:
        st.info("Upload or paste a statement to begin.")


# ============================================================
# TAB 2: Rules
# ============================================================
with nav[1]:
    st.subheader("Rules")
    st.caption("Edit matching rules. Higher priority rules at the top. First match wins.")

    # Save/Load + rules downloads
    with st.expander("Save / Load Session (keeps your review edits)", expanded=False):
        colA, colB = st.columns(2)
        with colA:
            if st.button("Save session into memory", key="save_session_mem"):
                st.session_state["session_snapshot"] = export_session_payload()
                st.success("Saved snapshot (in this Streamlit session).")
        with colB:
            if st.button("Restore snapshot from memory", key="restore_session_mem"):
                snap = st.session_state.get("session_snapshot")
                if snap:
                    df_raw, df_enriched, df_review = import_session_payload(snap)
                    if df_raw is not None:
                        st.session_state["df_raw"] = df_raw
                    if df_enriched is not None:
                        st.session_state["df_enriched"] = df_enriched
                    if df_review is not None:
                        st.session_state["df_review"] = df_review
                    st.success("Restored snapshot.")
                else:
                    st.warning("No snapshot saved yet.")

        payload = export_session_payload()
        payload_bytes = json.dumps(payload, indent=2).encode("utf-8")
        st.download_button(
            "Download session file (JSON)",
            data=payload_bytes,
            file_name=f"tax_buckets_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

        uploaded_session = st.file_uploader("Load session file (JSON)", type=["json"], key="session_loader")
        if uploaded_session is not None:
            try:
                loaded = json.loads(uploaded_session.read().decode("utf-8"))
                df_raw, df_enriched, df_review = import_session_payload(loaded)
                if df_raw is not None:
                    st.session_state["df_raw"] = df_raw
                if df_enriched is not None:
                    st.session_state["df_enriched"] = df_enriched
                if df_review is not None:
                    st.session_state["df_review"] = df_review
                st.success("Session loaded (rules + data + review).")
            except Exception as e:
                st.error(f"Could not load session JSON: {e}")

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        rules_df = pd.DataFrame(st.session_state["rules"])
        edited_rules = st.data_editor(
            rules_df,
            num_rows="dynamic",
            width='stretch',
            key="rules_editor",
        )

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            if st.button("Save rules changes", key="save_rules_btn"):
                st.session_state["rules"] = edited_rules.to_dict(orient="records")
                st.success("Rules saved.")
        with c2:
            if st.button("Reset rules to defaults", key="reset_rules_btn"):
                st.session_state["rules"] = default_rules()
                st.success("Rules reset.")
        with c3:
            rules_only_bytes = json.dumps(st.session_state["rules"], indent=2).encode("utf-8")
            st.download_button("Download rules.json", data=rules_only_bytes, file_name="rules.json", mime="application/json")

        uploaded_rules = st.file_uploader("Load rules.json", type=["json"], key="rules_loader")
        if uploaded_rules is not None:
            try:
                loaded_rules = json.loads(uploaded_rules.read().decode("utf-8"))
                if isinstance(loaded_rules, list):
                    st.session_state["rules"] = loaded_rules
                    st.success("Rules loaded.")
                else:
                    st.error("rules.json must be a list of rule objects.")
            except Exception as e:
                st.error(f"Could not load rules.json: {e}")

    with col2:
        st.markdown("#### Notes")
        st.write(
            "- Put **specific merchants** above generic ones.\n"
            "- You can keep adding rules as you review.\n"
            "- Assistant pay defaults for outgoing Venmo/PayPal/Zelle.\n"
            "- Incoming payment apps default to **Business Income** (change if needed)."
        )


# ============================================================
# TAB 3: Review + Export
# ============================================================
with nav[2]:
    st.subheader("Review + Export")

    if "df_raw" not in st.session_state or not isinstance(st.session_state["df_raw"], pd.DataFrame) or st.session_state["df_raw"].empty:
        st.warning("Import a statement first (Import / Paste tab).")
        st.stop()

    # Build enriched once
    if "df_enriched" not in st.session_state or st.session_state["df_enriched"] is None or st.session_state["df_enriched"].empty:
        st.session_state["df_enriched"] = enrich_transactions(st.session_state["df_raw"], rules=st.session_state["rules"])

    if "df_review" not in st.session_state or st.session_state["df_review"] is None or st.session_state["df_review"].empty:
        st.session_state["df_review"] = st.session_state["df_enriched"].copy()

    # Filters
    col1, col2, col3 = st.columns([1.2, 1.2, 1.6])
    with col1:
        show_needs_review_only = st.checkbox("Needs Review only", value=False)
    with col2:
        show_assistant_pay_only = st.checkbox("Assistant Pay only", value=False)
    with col3:
        search = st.text_input("Search description/merchant", value="")

    view_df = st.session_state["df_review"].copy()
    view_df = ensure_cols(view_df, ["Category_Manual"], default="")

    if show_needs_review_only:
        view_df = view_df[view_df["Bucket"] == "Uncategorized / Needs Review"]
    if show_assistant_pay_only:
        view_df = view_df[view_df["Category"] == "Assistant Pay"]
    if search.strip():
        s = search.strip().lower()
        view_df = view_df[
            view_df["Merchant"].astype(str).str.lower().str.contains(s, na=False)
            | view_df["Original Description"].astype(str).str.lower().str.contains(s, na=False)
        ]

    # Smarter category dropdown: defaults + what you already used + loan cats
    existing_cats = (
        st.session_state["df_review"]["Category"].dropna().astype(str).str.strip().tolist()
        if "df_review" in st.session_state and isinstance(st.session_state["df_review"], pd.DataFrame)
        else []
    )
    all_category_options = sorted(set([c for c in existing_cats if c] + DEFAULT_CATEGORIES + ["Loan", "Loan: SBA EIDL"]))

    st.caption(
        "Edit Bucket/Category/Subcategory/Project/Assistant/Notes. "
        "If the dropdown doesn’t include what you need, type it in **Category_Manual** (it overrides Category). "
        "Then click **Save Review Edits**."
    )

    edited = st.data_editor(
        view_df,
        width='stretch',
        num_rows="dynamic",
        column_config={
            "Bucket": st.column_config.SelectboxColumn("Bucket", options=DEFAULT_BUCKETS, required=False),
            "Category": st.column_config.SelectboxColumn("Category", options=all_category_options, required=False),
            "Category_Manual": st.column_config.TextColumn(
                "Category (manual override)",
                help="Type any category here if it’s not in the dropdown. This overrides Category.",
                required=False,
            ),
            "Confidence": st.column_config.NumberColumn(format="%.2f"),
            "Has FX": st.column_config.CheckboxColumn(),
        },
        key="review_editor",
    )

    # apply manual override immediately in-memory so it saves correctly
    edited = apply_manual_override(edited, manual_col="Category_Manual", target_col="Category")

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("Save Review Edits", key="save_review_btn"):
            base = st.session_state["df_review"].copy()
            base = ensure_cols(base, ["Category_Manual"], default="")

            def make_key(df: pd.DataFrame) -> pd.Series:
                return (
                    df["Account"].astype(str)
                    + "|"
                    + df["Section"].astype(str)
                    + "|"
                    + df["Transaction date"].astype(str)
                    + "|"
                    + df["Amount"].astype(str)
                    + "|"
                    + df["Original Description"].astype(str)
                )

            base["_k"] = make_key(base)

            edited2 = edited.copy()
            edited2 = ensure_cols(edited2, ["Category_Manual"], default="")
            edited2 = apply_manual_override(edited2, manual_col="Category_Manual", target_col="Category")
            edited2["_k"] = make_key(edited2)

            edited_map = edited2.set_index("_k")
            base_map = base.set_index("_k")

            up_cols = ["Bucket", "Category", "Subcategory", "Project", "Assistant", "Notes", "Merchant", "Category_Manual"]

            common_idx = base_map.index.intersection(edited_map.index)
            if len(common_idx) == 0:
                st.info("No matching rows to update.")
            else:
                for col in up_cols:
                    if col in edited_map.columns:
                        base_map.loc[common_idx, col] = edited_map.loc[common_idx, col]

                base_out = base_map.reset_index().drop(columns=["_k"])
                st.session_state["df_review"] = dedupe_transactions(base_out)
                st.success("Saved! Export will now use your edited categories.")

    with c2:
        if st.button("Re-apply rules to blank/Needs Review", key="reapply_rules_btn"):
            d = st.session_state["df_review"].copy()
            d = ensure_cols(d, ["Category_Manual"], default="")
            mask = (d["Bucket"].astype(str).str.strip() == "") | (d["Bucket"] == "Uncategorized / Needs Review")
            if mask.any():
                temp = enrich_transactions(d[mask].copy(), rules=st.session_state["rules"])
                d.loc[mask, temp.columns] = temp.values
                st.session_state["df_review"] = dedupe_transactions(d)
                st.success("Rules re-applied to rows that were blank/Needs Review.")
            else:
                st.info("Nothing to re-apply.")

    with c3:
        st.write(f"Rows shown: **{len(view_df)}** | Total saved: **{len(st.session_state['df_review'])}**")

    st.divider()
    st.markdown("### Export")
    expenses_negative = st.checkbox("QuickBooks bank feed: expenses negative", value=True)
    final_df = dedupe_transactions(st.session_state["df_review"].copy())

    colE1, colE2 = st.columns(2)

    with colE1:
        full_csv = final_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download FULL CSV", data=full_csv, file_name="FULL_export.csv", mime="text/csv")

        full_xlsx = df_to_excel_bytes(final_df, sheet_name="Full Export")
        st.download_button(
            "Download FULL XLSX",
            data=full_xlsx,
            file_name="FULL_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with colE2:
        qb_df = to_quickbooks_bank_feed(final_df, expenses_negative=expenses_negative)
        qb_csv = qb_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download QB Bank Feed CSV", data=qb_csv, file_name="QB_bank_feed.csv", mime="text/csv")

        qb_xlsx = df_to_excel_bytes(qb_df, sheet_name="QB Bank Feed")
        st.download_button(
            "Download QB Bank Feed XLSX",
            data=qb_xlsx,
            file_name="QB_bank_feed.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with st.expander("Preview (QB bank feed)", expanded=False):
        st.dataframe(qb_df.head(100), width='stretch')


# ============================================================
# TAB 4: Reports
# ============================================================
with nav[3]:
    st.subheader("Reports")

    if "df_review" not in st.session_state or not isinstance(st.session_state["df_review"], pd.DataFrame) or st.session_state["df_review"].empty:
        st.info("Import + Review first. Reports will appear here.")
        st.stop()

    d = st.session_state["df_review"].copy()
    d["Amount_num"] = d["Amount"].apply(lambda x: float(x) if pd.notna(x) else 0.0)

    st.markdown("### Totals by Bucket / Category (absolute spend)")
    pivot = (
        d.assign(AbsAmount=d["Amount_num"].abs())
        .groupby(["Bucket", "Category"], dropna=False)["AbsAmount"]
        .sum()
        .reset_index()
        .sort_values("AbsAmount", ascending=False)
    )
    st.dataframe(pivot, width='stretch')

    st.markdown("### Top Merchants (absolute spend)")
    merch = (
        d.assign(AbsAmount=d["Amount_num"].abs())
        .groupby(["Merchant"], dropna=False)["AbsAmount"]
        .sum()
        .reset_index()
        .sort_values("AbsAmount", ascending=False)
        .head(30)
    )
    st.dataframe(merch, width='stretch')

    st.markdown("### FX fees")
    fx = d[d.get("Has FX", False) == True].copy()
    fx_fees = d[d["Original Description"].str.contains("Foreign Exch Rt ADJ Fee", case=False, na=False)].copy()
    col1, col2, col3 = st.columns(3)
    col1.metric("FX transactions (count)", int(len(fx)))
    col2.metric("FX fee lines (count)", int(len(fx_fees)))
    col3.metric("FX fees total", f"${fx_fees['Amount_num'].abs().sum():,.2f}")

    st.markdown("### Assistant Pay totals")
    ap = d[d["Category"] == "Assistant Pay"].copy()
    st.write(f"Assistant Pay rows: **{len(ap)}**")
    st.write(f"Assistant Pay total (abs): **${ap['Amount_num'].abs().sum():,.2f}**")
    if len(ap) > 0:
        st.dataframe(
            ap[["Transaction date", "Merchant", "Original Description", "Amount", "Assistant", "Notes"]].head(200),
            width='stretch',
        )

    with st.expander("Optional: rules snapshot (hidden by default)", expanded=False):
        st.json(st.session_state.get("rules", []))

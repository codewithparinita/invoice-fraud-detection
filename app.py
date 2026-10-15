
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import IsolationForest
from supabase import create_client
from datetime import datetime, timezone

import io

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


st.set_page_config(
    page_title="Invoice Fraud & Vendor Risk AI",
    page_icon="🔎",
    layout="wide"
)

REQUIRED = [
    "invoice_id", "vendor_id", "invoice_date",
    "amount", "bank_account"
]


# =====================================================
# 1. SUPABASE CONNECTION
# =====================================================

@st.cache_resource
def create_database_client(url, key):
    return create_client(url, key)


def connect_database():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        if not url or not key:
            return None
        return create_database_client(url, key)
    except Exception:
        return None


db = connect_database()


# =====================================================
# 2. GENERATE SAMPLE DATA
# =====================================================

def make_demo_data(n=600, seed=42):
    rng = np.random.default_rng(seed)

    vendors = [f"V{i:03d}" for i in range(1, 31)]
    accounts = {
        v: f"ACC{i:04d}"
        for i, v in enumerate(vendors)
    }

    df = pd.DataFrame({
        "invoice_id": [f"INV{i:05d}" for i in range(n)],
        "vendor_id": rng.choice(vendors, n),
        "invoice_date": (
            pd.Timestamp("2025-01-01")
            + pd.to_timedelta(
                rng.integers(0, 365, n), unit="D"
            )
        ),
        "amount": np.round(
            rng.lognormal(8.5, 0.8, n), 2
        ),
        "payment_days": rng.integers(1, 61, n)
    })

    df["bank_account"] = df["vendor_id"].map(accounts)
    df["synthetic_label"] = 0

    suspicious = rng.choice(df.index, 20, replace=False)
    df.loc[suspicious, "amount"] = rng.uniform(
        50000, 120000, len(suspicious)
    ).round(2)
    df.loc[suspicious, "synthetic_label"] = 1

    duplicates = rng.choice(df.index, 8, replace=False)
    for idx in duplicates:
        source = max(0, idx - 1)
        df.loc[idx, "invoice_id"] = df.loc[
            source, "invoice_id"
        ]
        df.loc[idx, "synthetic_label"] = 1

    changed = rng.choice(df.index, 8, replace=False)
    df.loc[changed, "bank_account"] = [
        f"CHANGED{i}" for i in changed
    ]
    df.loc[changed, "synthetic_label"] = 1

    df["invoice_date"] = df["invoice_date"].dt.strftime(
        "%Y-%m-%d"
    )

    return df


# =====================================================
# 3. FETCH VERIFIED VENDOR RECORDS
# =====================================================

def get_vendor_records(client, vendor_ids):
    if client is None:
        return {}

    ids = list({
        str(v).strip()
        for v in vendor_ids
        if pd.notna(v) and str(v).strip()
    })

    if not ids:
        return {}

    response = (
        client.table("vendors")
        .select(
            "vendor_id, vendor_name, verified_bank_account"
        )
        .in_("vendor_id", ids)
        .execute()
    )

    return {
        str(row["vendor_id"]).strip(): row
        for row in (response.data or [])
    }


# =====================================================
# 4. INVOICE FRAUD ANALYSIS
# =====================================================

def analyze_invoices(raw, client=None):
    df = raw.copy()

    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(
            "Missing columns: " + ", ".join(sorted(missing))
        )

    for col in ["invoice_id", "vendor_id", "bank_account"]:
        df[col] = df[col].astype("string").str.strip()

    df["amount"] = pd.to_numeric(
        df["amount"], errors="coerce"
    )

    df["invoice_date"] = pd.to_datetime(
        df["invoice_date"], errors="coerce"
    )

    if df[REQUIRED].isnull().any().any():
        raise ValueError(
            "Required fields contain missing or invalid values."
        )

    if (
        df[["invoice_id", "vendor_id", "bank_account"]] == ""
    ).any().any():
        raise ValueError(
            "Invoice ID, vendor ID and bank account cannot be blank."
        )

    if (df["amount"] <= 0).any():
        raise ValueError("Invoice amounts must be positive.")

    # Rule 1: Duplicate invoice IDs
    df["duplicate_invoice"] = df.duplicated(
        subset=["invoice_id"], keep=False
    )

    # Rule 2: Unusually high amounts
    median = df.groupby("vendor_id")["amount"].transform("median")
    df["unusual_amount"] = (
        df["amount"] > np.maximum(median * 3, 10000)
    )

    # Rule 3: Account differences inside this upload
    expected = (
        df.groupby("vendor_id")["bank_account"]
        .agg(lambda values: values.mode().iloc[0])
    )

    df["account_mismatch"] = df.apply(
        lambda row: (
            row["bank_account"] != expected[row["vendor_id"]]
        ),
        axis=1
    )

    # Rule 4: Compare against verified vendor master
    records = get_vendor_records(client, df["vendor_id"])

    df["vendor_registered"] = df["vendor_id"].map(
        lambda v: str(v).strip() in records
    )

    df["verified_bank_account"] = df["vendor_id"].map(
        lambda v: records.get(
            str(v).strip(), {}
        ).get("verified_bank_account", "")
    ).fillna("").astype(str).str.strip()

    df["vendor_bank_mismatch"] = df.apply(
        lambda row: (
            bool(row["vendor_registered"])
            and bool(row["verified_bank_account"])
            and row["bank_account"] != row["verified_bank_account"]
        ),
        axis=1
    )

    # Rule 5: Isolation Forest
    features = pd.DataFrame({
        "log_amount": np.log1p(df["amount"])
    })

    if "payment_days" in df.columns:
        payment = pd.to_numeric(
            df["payment_days"], errors="coerce"
        )
        payment_median = payment.median()
        if pd.isna(payment_median):
            payment_median = 0
        features["payment_days"] = payment.fillna(payment_median)

    if len(df) >= 10 and features.nunique().sum() > 1:
        model = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=42
        )
        df["ml_anomaly"] = model.fit_predict(features) == -1
    else:
        df["ml_anomaly"] = False

    # Risk score is a prioritization score, not fraud probability
    df["risk_score"] = (
        df["duplicate_invoice"].astype(int) * 40
        + df["unusual_amount"].astype(int) * 25
        + df["account_mismatch"].astype(int) * 15
        + df["vendor_bank_mismatch"].astype(int) * 40
        + df["ml_anomaly"].astype(int) * 15
    ).clip(upper=100)

    df["risk_level"] = pd.cut(
        df["risk_score"],
        bins=[-1, 24, 49, 100],
        labels=["Low", "Medium", "High"]
    ).astype(str)

    def reasons(row):
        items = []

        if row["duplicate_invoice"]:
            items.append("Duplicate invoice ID")
        if row["unusual_amount"]:
            items.append("Unusually high invoice amount")
        if row["account_mismatch"]:
            items.append("Account differs from batch pattern")
        if row["vendor_bank_mismatch"]:
            items.append(
                "Bank account differs from verified vendor record"
            )
        if client is not None and not row["vendor_registered"]:
            items.append("Vendor not registered in vendor master")
        if row["ml_anomaly"]:
            items.append("Isolation Forest anomaly")

        return "; ".join(items) if items else "No rule triggered"

    df["risk_reasons"] = df.apply(reasons, axis=1)

    return df


# =====================================================
# 5. SAVE INVESTIGATION HISTORY
# =====================================================

def save_investigations(client, results):
    if client is None:
        raise RuntimeError("Supabase is not connected.")

    flagged = results[results["risk_score"] > 0]

    if flagged.empty:
        return 0

    rows = []

    for _, row in flagged.iterrows():
        rows.append({
            "invoice_id": str(row["invoice_id"]),
            "vendor_id": str(row["vendor_id"]),
            "detected_issue": str(row["risk_reasons"])[:1000],
            "risk_score": float(row["risk_score"])
        })

    client.table("investigation_history").insert(
        rows
    ).execute()

    return len(rows)


# =====================================================
# INVESTIGATION REPORT FUNCTIONS
# =====================================================

def load_investigation_report_data():
    """Load saved investigation records from Supabase."""
    if db is None:
        return pd.DataFrame()

    response = (
        db.table("investigation_history")
        .select("*")
        .order("investigated_at", desc=True)
        .execute()
    )

    records = response.data or []
    df = pd.DataFrame(records)

    if df.empty:
        return df

    # Retrieve vendor names from the vendor master.
    vendor_response = (
        db.table("vendors")
        .select("vendor_id, vendor_name")
        .execute()
    )

    vendor_names = {
        row["vendor_id"]: row["vendor_name"]
        for row in (vendor_response.data or [])
    }

    df["vendor_name"] = (
        df["vendor_id"].map(vendor_names).fillna("Not available")
    )

    df["risk_score"] = pd.to_numeric(
        df["risk_score"], errors="coerce"
    )

    def get_risk_level(score):
        if pd.isna(score):
            return "Unscored"
        if score >= 50:
            return "High"
        if score >= 25:
            return "Medium"
        return "Low"

    df["risk_level"] = df["risk_score"].apply(get_risk_level)

    return df


def create_investigation_pdf(df):
    """Create a downloadable PDF investigation report."""
    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=10 * mm,
        leftMargin=10 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm
    )

    styles = getSampleStyleSheet()
    elements = []

    elements.append(
        Paragraph(
            "AI-Based Invoice Fraud &amp; Vendor Risk Detection",
            styles["Title"]
        )
    )
    elements.append(
        Paragraph("Investigation Report", styles["Heading2"])
    )
    elements.append(
        Paragraph(
            "Generated: " + datetime.now().strftime("%d-%m-%Y %H:%M"),
            styles["Normal"]
        )
    )
    elements.append(Spacer(1, 10))

    total = len(df)
    high = int((df["risk_level"] == "High").sum())
    medium = int((df["risk_level"] == "Medium").sum())
    low = int((df["risk_level"] == "Low").sum())

    elements.append(
        Paragraph(
            f"Total records: {total} | High risk: {high} | "
            f"Medium risk: {medium} | Low risk: {low}",
            styles["Normal"]
        )
    )
    elements.append(Spacer(1, 12))

    columns = [
        ("invoice_id", "Invoice ID"),
        ("vendor_id", "Vendor ID"),
        ("vendor_name", "Vendor Name"),
        ("detected_issue", "Detected Issue"),
        ("risk_score", "Risk Score"),
        ("risk_level", "Risk Level"),
        ("investigated_at", "Investigation Date")
    ]

    table_data = [[heading for _, heading in columns]]

    for _, row in df.iterrows():
        table_row = []

        for key, _ in columns:
            value = row.get(key, "")

            if pd.isna(value):
                value = ""
            elif key == "risk_score":
                value = f"{float(value):.2f}"
            else:
                value = str(value)

            value = (
                value.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

            table_row.append(
                Paragraph(value, styles["BodyText"])
            )

        table_data.append(table_row)

    table = Table(
        table_data,
        colWidths=[
            25 * mm, 22 * mm, 32 * mm, 62 * mm,
            20 * mm, 22 * mm, 35 * mm
        ],
        repeatRows=1
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B53")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F1F5F9")]),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 10))
    elements.append(
        Paragraph(
            "A risk flag indicates a need for investigation; it does "
            "not prove that fraud occurred.",
            styles["Italic"]
        )
    )

    document.build(elements)
    buffer.seek(0)
    return buffer.getvalue()



# =====================================================
# 6. APP NAVIGATION
# =====================================================

st.title("🔎 AI-Based Invoice Fraud & Vendor Risk Detection")

st.caption(
    "Final-year project | Machine learning, vendor verification, "
    "risk analytics and investigation history"
)

with st.sidebar:
    st.header("Navigation")

    page = st.radio(
        "Choose page",
        [
            "Overview",
            "Analyze Invoices",
            "Vendor Risk",
            "Vendor Master",
            "Investigation History",
            "Historical Vendor Risk",
            "Investigation Reports",
            "Fraud Alerts",
            "Methodology"
        ]
    )

    st.divider()

    if db is not None:
        st.success("Supabase connected")
    else:
        st.warning("Supabase not connected")

    st.caption("Use synthetic or authorized data only.")

if "demo_df" not in st.session_state:
    st.session_state.demo_df = None

if "results" not in st.session_state:
    st.session_state.results = None


# =====================================================
# 7. OVERVIEW
# =====================================================

if page == "Overview":
    st.subheader("Project Overview")

    st.write(
        "Analyze invoices, detect suspicious patterns, compare "
        "vendor bank accounts with verified records, and review "
        "historical investigations."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Detection Methods", "5")
    c2.metric("Input Format", "CSV")
    c3.metric(
        "Database",
        "Connected" if db is not None else "Not connected"
    )

    st.markdown("""
    **Main features**
    - Duplicate invoice detection
    - Unusually high amount detection
    - Batch bank-account inconsistency detection
    - Verified vendor bank-account comparison
    - Isolation Forest anomaly detection
    - Transparent risk scoring
    - Vendor analytics and visualizations
    - Persistent investigation history
    - Historical vendor risk dashboard
    - CSV report export
    """)

    if db is None:
        st.warning(
            "Configure SUPABASE_URL and SUPABASE_KEY in Streamlit "
            "Secrets to enable database features."
        )


# =====================================================
# 8. ANALYZE INVOICES
# =====================================================

elif page == "Analyze Invoices":
    st.subheader("Upload or Generate Invoice Data")

    if st.button("Generate Sample Invoices"):
        st.session_state.demo_df = make_demo_data()
        st.session_state.results = None
        st.success("Sample invoices generated.")

    uploaded = st.file_uploader(
        "Upload invoice CSV", type=["csv"]
    )

    source = None

    if uploaded is not None:
        try:
            source = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
    elif st.session_state.demo_df is not None:
        source = st.session_state.demo_df.copy()

    if source is not None:
        st.markdown("### Input Preview")
        st.dataframe(source.head(10), use_container_width=True)
        st.caption("Required columns: " + ", ".join(REQUIRED))

        if st.button("Run AI Fraud Analysis", type="primary"):
            try:
                st.session_state.results = analyze_invoices(source, db)
                st.success("Analysis completed.")
            except Exception as exc:
                st.session_state.results = None
                st.error(f"Analysis failed: {exc}")

    results = st.session_state.results

    if results is not None:
        st.divider()
        st.subheader("Analysis Results")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Invoices", len(results))
        c2.metric(
            "High Risk",
            int((results["risk_level"] == "High").sum())
        )
        c3.metric(
            "Medium Risk",
            int((results["risk_level"] == "Medium").sum())
        )
        c4.metric("Vendors", results["vendor_id"].nunique())

        st.dataframe(
            results.sort_values("risk_score", ascending=False),
            use_container_width=True
        )

        st.download_button(
            "Download Full Analysis CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="invoice_fraud_analysis.csv",
            mime="text/csv"
        )

        if db is not None:
            if st.button("Save Suspicious Invoices to Database"):
                try:
                    count = save_investigations(db, results)
                    st.success(f"Saved {count} investigation entries.")
                except Exception as exc:
                    st.error(f"Could not save records: {exc}")
        else:
            st.info("Connect Supabase to save investigation history.")

        st.warning(
            "A risk flag is not proof of fraud. Review evidence "
            "before making payment or vendor decisions."
        )


# =====================================================
# 9. CURRENT-DATA VENDOR RISK
# =====================================================

elif page == "Vendor Risk":
    st.subheader("Vendor Risk Analytics")

    results = st.session_state.results

    if results is None:
        st.info("Open Analyze Invoices and run an analysis first.")
    else:
        vendors = results.groupby("vendor_id").agg(
            invoice_count=("invoice_id", "size"),
            total_invoice_value=("amount", "sum"),
            average_invoice_value=("amount", "mean"),
            maximum_risk_score=("risk_score", "max"),
            high_risk_invoices=(
                "risk_level", lambda values: (values == "High").sum()
            ),
            account_mismatches=("account_mismatch", "sum"),
            verified_account_mismatches=(
                "vendor_bank_mismatch", "sum"
            )
        ).reset_index()

        vendors = vendors.sort_values(
            ["high_risk_invoices", "maximum_risk_score"],
            ascending=False
        )

        st.markdown("### Vendor Summary")
        st.dataframe(vendors, use_container_width=True)

        fig = px.bar(
            vendors.head(15),
            x="vendor_id",
            y="total_invoice_value",
            title="Invoice Value by Vendor"
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(
            vendors,
            x="invoice_count",
            y="maximum_risk_score",
            size="total_invoice_value",
            hover_name="vendor_id",
            title="Vendor Activity vs Maximum Risk Score"
        )
        st.plotly_chart(fig2, use_container_width=True)


# =====================================================
# 10. VENDOR MASTER
# =====================================================

elif page == "Vendor Master":
    st.subheader("Verified Vendor Master Management")

    if db is None:
        st.error("Supabase is not connected.")
    else:
        with st.form("vendor_form"):
            vendor_id = st.text_input("Vendor ID").strip()
            vendor_name = st.text_input("Vendor Name").strip()
            bank_account = st.text_input(
                "Verified Bank Account"
            ).strip()

            verified = st.checkbox(
                "I have independently verified these details."
            )

            submitted = st.form_submit_button(
                "Register / Update Vendor"
            )

        if submitted:
            if not vendor_id or not vendor_name or not bank_account:
                st.error("Complete all fields.")
            elif not verified:
                st.error("Confirm that you verified the vendor details.")
            else:
                try:
                    old = (
                        db.table("vendors")
                        .select("verified_bank_account")
                        .eq("vendor_id", vendor_id)
                        .execute()
                    )

                    if old.data:
                        old_account = str(
                            old.data[0]["verified_bank_account"]
                        ).strip()

                        if old_account != bank_account:
                            st.warning(
                                "The verified account is being changed. "
                                "Ensure the new account has been checked."
                            )

                    db.table("vendors").upsert(
                        {
                            "vendor_id": vendor_id,
                            "vendor_name": vendor_name,
                            "verified_bank_account": bank_account,
                            "updated_at": datetime.now(
                                timezone.utc
                            ).isoformat()
                        },
                        on_conflict="vendor_id"
                    ).execute()

                    st.success("Vendor saved successfully.")
                except Exception as exc:
                    st.error(f"Could not save vendor: {exc}")

        st.markdown("### Registered Vendors")

        try:
            response = (
                db.table("vendors")
                .select(
                    "vendor_id, vendor_name, verified_bank_account, "
                    "registered_at, updated_at"
                )
                .order("vendor_id")
                .execute()
            )

            vendor_data = response.data or []

            if vendor_data:
                st.dataframe(
                    pd.DataFrame(vendor_data),
                    use_container_width=True
                )
            else:
                st.info("No vendors registered yet.")
        except Exception as exc:
            st.error(f"Could not load vendors: {exc}")


# =====================================================
# 11. INVESTIGATION HISTORY
# =====================================================

elif page == "Investigation History":
    st.subheader("Persistent Investigation History")

    if db is None:
        st.error("Supabase is not connected.")
    else:
        try:
            response = (
                db.table("investigation_history")
                .select(
                    "id, invoice_id, vendor_id, detected_issue, "
                    "risk_score, investigated_at"
                )
                .order("investigated_at", desc=True)
                .limit(1000)
                .execute()
            )

            history = response.data or []

            if history:
                history_df = pd.DataFrame(history)

                c1, c2, c3 = st.columns(3)
                c1.metric("Saved Entries", len(history_df))
                c2.metric(
                    "Invoices",
                    history_df["invoice_id"].nunique()
                )
                c3.metric(
                    "Vendors",
                    history_df["vendor_id"].nunique()
                )

                st.dataframe(
                    history_df,
                    use_container_width=True
                )

                st.download_button(
                    "Download Investigation History",
                    data=history_df.to_csv(index=False).encode("utf-8"),
                    file_name="investigation_history.csv",
                    mime="text/csv"
                )
            else:
                st.info(
                    "No records found. Run invoice analysis and save "
                    "the suspicious findings first."
                )
        except Exception as exc:
            st.error(f"Could not load history: {exc}")


# =====================================================
# 12. HISTORICAL VENDOR RISK DASHBOARD
# =====================================================

elif page == "Historical Vendor Risk":
    st.subheader("Historical Vendor Risk Dashboard")

    if db is None:
        st.error("Supabase is not connected.")
    else:
        try:
            response = (
                db.table("investigation_history")
                .select(
                    "vendor_id, invoice_id, risk_score, "
                    "detected_issue, investigated_at"
                )
                .order("investigated_at", desc=True)
                .limit(5000)
                .execute()
            )

            history = response.data or []

            if not history:
                st.info(
                    "No saved investigations yet. Analyze invoices "
                    "and save the findings first."
                )
            else:
                history_df = pd.DataFrame(history)

                history_df["risk_score"] = pd.to_numeric(
                    history_df["risk_score"], errors="coerce"
                ).fillna(0)

                vendor_risk = history_df.groupby("vendor_id").agg(
                    investigated_invoices=("invoice_id", "nunique"),
                    investigation_entries=("invoice_id", "count"),
                    average_recorded_score=("risk_score", "mean"),
                    maximum_recorded_score=("risk_score", "max")
                ).reset_index()

                vendor_risk["historical_risk_rating"] = (
                    vendor_risk["maximum_recorded_score"].apply(
                        lambda score:
                        "High" if score >= 50
                        else "Medium" if score >= 25
                        else "Low"
                    )
                )

                vendor_risk = vendor_risk.sort_values(
                    "maximum_recorded_score", ascending=False
                )

                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Vendors Investigated",
                    vendor_risk["vendor_id"].nunique()
                )
                c2.metric(
                    "High-Risk Vendors",
                    int(
                        (
                            vendor_risk["historical_risk_rating"]
                            == "High"
                        ).sum()
                    )
                )
                c3.metric("Investigation Entries", len(history_df))

                st.markdown("### Historical Vendor Ratings")
                st.dataframe(vendor_risk, use_container_width=True)

                fig = px.bar(
                    vendor_risk,
                    x="vendor_id",
                    y="maximum_recorded_score",
                    color="historical_risk_rating",
                    title="Maximum Recorded Investigation Score",
                    labels={
                        "vendor_id": "Vendor ID",
                        "maximum_recorded_score": "Risk Score",
                        "historical_risk_rating": "Historical Rating"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.scatter(
                    vendor_risk,
                    x="investigated_invoices",
                    y="average_recorded_score",
                    size="investigation_entries",
                    color="historical_risk_rating",
                    hover_name="vendor_id",
                    title="Investigated Invoices vs Average Recorded Score"
                )
                st.plotly_chart(fig2, use_container_width=True)

                st.download_button(
                    "Download Historical Vendor Risk Report",
                    data=vendor_risk.to_csv(index=False).encode("utf-8"),
                    file_name="historical_vendor_risk.csv",
                    mime="text/csv"
                )

                st.caption(
                    "Ratings are based only on saved flagged records. "
                    "They are not a fraud probability or a complete "
                    "measure of vendor reliability."
                )

        except Exception as exc:
            st.error(f"Could not load historical risk: {exc}")


elif page == "Investigation Reports":
    st.title("📄 Investigation Reports")
    st.write(
        "View saved investigation records and download reports."
    )

    try:
        report_df = load_investigation_report_data()

        if report_df.empty:
            st.info(
                "No saved investigations found. First analyze invoices "
                "and save suspicious invoices to the database."
            )
        else:
            col1, col2, col3 = st.columns(3)

            col1.metric("Total Records", len(report_df))
            col2.metric(
                "High Risk",
                int((report_df["risk_level"] == "High").sum())
            )
            col3.metric(
                "Medium Risk",
                int((report_df["risk_level"] == "Medium").sum())
            )

            st.subheader("Investigation Records")

            display_columns = [
                "invoice_id",
                "vendor_id",
                "vendor_name",
                "detected_issue",
                "risk_score",
                "risk_level",
                "investigated_at"
            ]

            st.dataframe(
                report_df[display_columns],
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download CSV Report",
                data=report_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="invoice_investigation_report.csv",
                mime="text/csv",
                use_container_width=True
            )

            pdf_data = create_investigation_pdf(report_df)

            st.download_button(
                "⬇️ Download PDF Report",
                data=pdf_data,
                file_name="invoice_investigation_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    except Exception as e:
        st.error("Could not generate the investigation report.")
        st.exception(e)

elif page == "Fraud Alerts":
    st.title("🚨 High-Risk Invoice Alerts")

    st.write(
        "Review saved investigations that have high recorded "
        "risk scores and require priority attention."
    )

    try:
        alerts_df = load_investigation_report_data()

        if alerts_df.empty:
            st.info(
                "No saved investigation records are available yet. "
                "Analyze invoices and save suspicious records first."
            )

        else:
            # High risk means a recorded score of 50 or more.
            high_alerts = alerts_df[
                alerts_df["risk_score"] >= 50
            ].copy()

            total_records = len(alerts_df)
            total_alerts = len(high_alerts)

            col1, col2, col3 = st.columns(3)

            col1.metric("Saved Investigations", total_records)
            col2.metric("High-Risk Alerts", total_alerts)
            col3.metric(
                "Vendors Requiring Review",
                high_alerts["vendor_id"].nunique()
            )

            if high_alerts.empty:
                st.success(
                    "No high-risk records were found among the "
                    "saved investigations."
                )

            else:
                st.error(
                    f"{total_alerts} high-risk investigation record(s) "
                    "require priority review."
                )

                st.subheader("High-Risk Cases")

                alert_columns = [
                    "invoice_id",
                    "vendor_id",
                    "vendor_name",
                    "detected_issue",
                    "risk_score",
                    "risk_level",
                    "investigated_at"
                ]

                st.dataframe(
                    high_alerts[alert_columns],
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    label="Download High-Risk Alerts (CSV)",
                    data=high_alerts.to_csv(
                        index=False
                    ).encode("utf-8-sig"),
                    file_name="high_risk_invoice_alerts.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                st.warning(
                    "Auditor action: verify the invoice, vendor details, "
                    "supporting documents and payment account before "
                    "approving or rejecting a transaction."
                )

            st.caption(
                "Alert threshold: risk score >= 50. "
                "Alerts are based on saved investigation records, "
                "not confirmed fraud."
            )

    except Exception as e:
        st.error("Unable to load high-risk alerts.")
        st.exception(e)

# =====================================================
# 13. METHODOLOGY
# =====================================================

elif page == "Methodology":
    st.subheader("Detection Methodology")

    st.markdown("""
    **1. Data validation:** checks required fields, valid dates,
    positive amounts and missing values.

    **2. Duplicate detection:** identifies repeated invoice IDs.

    **3. Amount anomaly detection:** compares amounts with the
    typical amount for that vendor in the uploaded dataset.

    **4. Vendor master verification:** compares invoice bank accounts
    with verified vendor accounts stored in Supabase.

    **5. Batch account checks:** identifies account inconsistencies
    inside the uploaded dataset.

    **6. Machine learning:** Isolation Forest identifies unusual
    records using invoice amounts and optional payment timing.

    **7. Risk scoring:** combines rule indicators and ML flags into
    a transparent 0–100 prioritization score.

    **8. Vendor analytics:** summarizes invoice activity by vendor.

    **9. Investigation history:** stores flagged findings in Supabase
    when the user explicitly saves the analysis.

    **10. Historical risk analytics:** summarizes saved investigation
    records by vendor.
    """)

    st.warning(
        "The score is not a calibrated fraud probability. "
        "An anomaly requires investigation and is not proof of fraud."
    )

    st.markdown("### Project Limitations")

    st.markdown("""
    - Isolation Forest learns from the uploaded dataset.
    - Initial thresholds require evaluation using representative data.
    - Saving the same analysis more than once can create duplicate
      investigation entries.
    - Historical risk ratings cover saved flagged records only.
    - Vendor master records must be independently verified.
    """)

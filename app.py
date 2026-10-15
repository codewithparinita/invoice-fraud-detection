import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import IsolationForest
from supabase import create_client
from datetime import datetime, timezone

st.set_page_config(
    page_title="Invoice Fraud & Vendor Risk AI",
    page_icon="🔎",
    layout="wide"
)

REQUIRED = [
    "invoice_id",
    "vendor_id",
    "invoice_date",
    "amount",
    "bank_account"
]


# =========================================================
# 1. SUPABASE DATABASE CONNECTION
# =========================================================

@st.cache_resource
def create_supabase_client(url, key):
    return create_client(url, key)


def get_database():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]

        if not url or not key:
            return None

        return create_supabase_client(url, key)

    except Exception:
        return None


db = get_database()


# =========================================================
# 2. SAMPLE INVOICE DATA
# =========================================================

def make_demo_data(n=600, seed=42):
    rng = np.random.default_rng(seed)

    vendors = [f"V{i:03d}" for i in range(1, 31)]
    accounts = {
        vendor: f"ACC{i:04d}"
        for i, vendor in enumerate(vendors)
    }

    df = pd.DataFrame({
        "invoice_id": [f"INV{i:05d}" for i in range(n)],
        "vendor_id": rng.choice(vendors, n),
        "invoice_date": (
            pd.Timestamp("2025-01-01")
            + pd.to_timedelta(
                rng.integers(0, 365, n),
                unit="D"
            )
        ),
        "amount": np.round(
            rng.lognormal(8.5, 0.8, n), 2
        ),
        "payment_days": rng.integers(1, 61, n)
    })

    df["bank_account"] = df["vendor_id"].map(accounts)
    df["synthetic_label"] = 0

    suspicious = rng.choice(
        df.index, size=20, replace=False
    )

    df.loc[suspicious, "amount"] = rng.uniform(
        50000, 120000, len(suspicious)
    ).round(2)

    df.loc[suspicious, "synthetic_label"] = 1

    dup = rng.choice(df.index, size=8, replace=False)

    for idx in dup:
        source = max(0, idx - 1)
        df.loc[idx, "invoice_id"] = df.loc[
            source, "invoice_id"
        ]
        df.loc[idx, "synthetic_label"] = 1

    changed = rng.choice(df.index, size=8, replace=False)

    df.loc[changed, "bank_account"] = [
        f"CHANGED{i}" for i in changed
    ]

    df.loc[changed, "synthetic_label"] = 1

    df["invoice_date"] = (
        df["invoice_date"].dt.strftime("%Y-%m-%d")
    )

    return df


# =========================================================
# 3. GET VERIFIED VENDOR RECORDS
# =========================================================

def get_vendor_records(client, vendor_ids):
    if client is None:
        return {}

    unique_ids = [
        str(v).strip()
        for v in vendor_ids
        if pd.notna(v) and str(v).strip()
    ]

    if not unique_ids:
        return {}

    response = (
        client.table("vendors")
        .select(
            "vendor_id, vendor_name, verified_bank_account"
        )
        .in_("vendor_id", list(set(unique_ids)))
        .execute()
    )

    records = {}

    for item in response.data or []:
        records[str(item["vendor_id"]).strip()] = item

    return records


# =========================================================
# 4. INVOICE ANALYSIS AND RISK SCORING
# =========================================================

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
        subset=["invoice_id"],
        keep=False
    )

    # Rule 2: Unusually high invoice amount
    vendor_median = df.groupby(
        "vendor_id"
    )["amount"].transform("median")

    df["unusual_amount"] = (
        df["amount"] > np.maximum(vendor_median * 3, 10000)
    )

    # Rule 3: Account inconsistency within uploaded batch
    expected_account = (
        df.groupby("vendor_id")["bank_account"]
        .agg(lambda s: s.mode().iloc[0])
    )

    df["account_mismatch"] = df.apply(
        lambda r: (
            r["bank_account"]
            != expected_account[r["vendor_id"]]
        ),
        axis=1
    )

    # Rule 4: Compare against trusted vendor master records
    vendor_records = get_vendor_records(
        client, df["vendor_id"].tolist()
    )

    df["vendor_registered"] = df["vendor_id"].map(
        lambda v: str(v).strip() in vendor_records
    )

    df["verified_bank_account"] = df["vendor_id"].map(
        lambda v: vendor_records.get(
            str(v).strip(), {}
        ).get("verified_bank_account", "")
    )

    df["verified_bank_account"] = (
        df["verified_bank_account"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["vendor_bank_mismatch"] = df.apply(
        lambda r: (
            bool(r["vendor_registered"])
            and bool(r["verified_bank_account"])
            and r["bank_account"] != r["verified_bank_account"]
        ),
        axis=1
    )

    # Rule 5: Unsupervised ML anomaly detection
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

        features["payment_days"] = payment.fillna(
            payment_median
        )

    if len(df) >= 10 and features.nunique().sum() > 1:
        model = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=42
        )

        df["ml_anomaly"] = (
            model.fit_predict(features) == -1
        )
    else:
        df["ml_anomaly"] = False

    # Transparent score: this is NOT a fraud probability.
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

    def get_reasons(row):
        reasons = []

        if row["duplicate_invoice"]:
            reasons.append("Duplicate invoice ID")

        if row["unusual_amount"]:
            reasons.append("Unusually high invoice amount")

        if row["account_mismatch"]:
            reasons.append(
                "Account differs from uploaded batch pattern"
            )

        if row["vendor_bank_mismatch"]:
            reasons.append(
                "Bank account differs from verified vendor record"
            )

        if (
            client is not None
            and not row["vendor_registered"]
        ):
            reasons.append(
                "Vendor not registered in vendor master"
            )

        if row["ml_anomaly"]:
            reasons.append("Isolation Forest anomaly")

        return "; ".join(reasons) if reasons else "No rule triggered"

    df["risk_reasons"] = df.apply(get_reasons, axis=1)

    return df


# =========================================================
# 5. SAVE INVESTIGATION RESULTS
# =========================================================

def save_investigations(client, results):
    if client is None:
        raise RuntimeError(
            "Supabase is not connected. Configure the secrets first."
        )

    suspicious = results[results["risk_score"] > 0]

    if suspicious.empty:
        return 0

    rows = []

    for _, row in suspicious.iterrows():
        rows.append({
            "invoice_id": str(row["invoice_id"]),
            "vendor_id": str(row["vendor_id"]),
            "detected_issue": str(row["risk_reasons"])[:1000],
            "risk_score": float(row["risk_score"])
        })

    # Insert only when the user explicitly clicks Save.
    # Clicking again creates a new investigation-history entry.
    client.table("investigation_history").insert(
        rows
    ).execute()

    return len(rows)


# =========================================================
# 6. PAGE HEADER AND NAVIGATION
# =========================================================

st.title("🔎 AI-Based Invoice Fraud & Vendor Risk Detection")

st.caption(
    "Final-year project | Machine learning, trusted vendor records, "
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
            "Methodology"
        ]
    )

    st.divider()

    if db is not None:
        st.success("Supabase connected")
    else:
        st.warning("Supabase not connected")

    st.caption("Use synthetic or authorized invoice data only.")

if "demo_df" not in st.session_state:
    st.session_state.demo_df = None

if "results" not in st.session_state:
    st.session_state.results = None


# =========================================================
# 7. OVERVIEW PAGE
# =========================================================

if page == "Overview":
    st.subheader("Project Overview")

    st.write(
        "This system analyzes invoice records, detects suspicious "
        "patterns, compares payment accounts with verified vendor "
        "records, and maintains an investigation history."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Detection methods", "5")
    c2.metric("Input format", "CSV")
    c3.metric(
        "Database",
        "Connected" if db is not None else "Not connected"
    )

    st.markdown("### Main features")

    st.markdown("""
    - Duplicate invoice detection
    - Unusually high invoice amount detection
    - Bank-account inconsistency checks
    - Verified vendor bank-account comparison
    - Isolation Forest machine learning
    - Transparent risk scoring
    - Vendor-level analytics and charts
    - Persistent investigation history
    - CSV report download
    """)

    if db is None:
        st.warning(
            "Configure SUPABASE_URL and SUPABASE_KEY in "
            "Streamlit secrets to enable persistent vendor records "
            "and investigation history."
        )


# =========================================================
# 8. ANALYZE INVOICES PAGE
# =========================================================

elif page == "Analyze Invoices":
    st.subheader("Upload or Generate Invoice Data")

    if db is None:
        st.info(
            "Analysis can still run without Supabase, but verified "
            "vendor comparisons and history saving require a connection."
        )

    if st.button("Generate Sample Invoices"):
        st.session_state.demo_df = make_demo_data()
        st.session_state.results = None
        st.success("Sample dataset generated.")

    uploaded = st.file_uploader(
        "Upload your invoice CSV",
        type=["csv"]
    )

    source = None

    if uploaded is not None:
        try:
            source = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    elif st.session_state.demo_df is not None:
        source = st.session_state.demo_df.copy()

    if source is not None:
        st.markdown("### Input Preview")

        st.dataframe(
            source.head(10),
            use_container_width=True
        )

        st.caption(
            "Required columns: " + ", ".join(REQUIRED)
        )

        if st.button(
            "Run AI Fraud Analysis",
            type="primary"
        ):
            try:
                st.session_state.results = analyze_invoices(
                    source, db
                )
                st.success("Analysis completed successfully.")

            except Exception as e:
                st.session_state.results = None
                st.error(f"Analysis failed: {e}")

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

        c4.metric(
            "Vendors",
            results["vendor_id"].nunique()
        )

        st.dataframe(
            results.sort_values(
                "risk_score", ascending=False
            ),
            use_container_width=True
        )

        st.download_button(
            "Download Full Analysis CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="invoice_fraud_analysis.csv",
            mime="text/csv"
        )

        st.markdown("### Save Investigation History")

        st.write(
            "Save flagged invoices to Supabase so the findings "
            "remain available after the app session ends."
        )

        if db is not None:
            if st.button("Save Suspicious Invoices to Database"):
                try:
                    saved_count = save_investigations(
                        db, results
                    )

                    st.success(
                        f"Saved {saved_count} investigation entries."
                    )

                except Exception as e:
                    st.error(
                        f"Could not save investigation history: {e}"
                    )
        else:
            st.info(
                "Connect Supabase to enable saving investigation history."
            )

        st.warning(
            "A high-risk flag is not proof of fraud. Review the "
            "supporting evidence before making payment or vendor decisions."
        )


# =========================================================
# 9. VENDOR RISK ANALYTICS PAGE
# =========================================================

elif page == "Vendor Risk":
    st.subheader("Vendor Risk Analytics")

    results = st.session_state.results

    if results is None:
        st.info("Analyze invoices first.")
    else:
        vendors = results.groupby("vendor_id").agg(
            invoice_count=("invoice_id", "size"),
            total_invoice_value=("amount", "sum"),
            average_invoice_value=("amount", "mean"),
            maximum_risk_score=("risk_score", "max"),
            high_risk_invoices=(
                "risk_level",
                lambda x: (x == "High").sum()
            ),
            account_mismatches=(
                "account_mismatch", "sum"
            ),
            verified_account_mismatches=(
                "vendor_bank_mismatch", "sum"
            )
        ).reset_index()

        vendors = vendors.sort_values(
            ["high_risk_invoices", "maximum_risk_score"],
            ascending=False
        )

        st.dataframe(
            vendors,
            use_container_width=True
        )

        fig = px.bar(
            vendors.head(15),
            x="vendor_id",
            y="total_invoice_value",
            title="Invoice Value by Vendor"
        )

        st.plotly_chart(
            fig, use_container_width=True
        )

        fig2 = px.scatter(
            vendors,
            x="invoice_count",
            y="maximum_risk_score",
            size="total_invoice_value",
            hover_name="vendor_id",
            title="Vendor Activity vs Maximum Risk Score"
        )

        st.plotly_chart(
            fig2, use_container_width=True
        )


# =========================================================
# 10. VENDOR MASTER MANAGEMENT PAGE
# =========================================================

elif page == "Vendor Master":
    st.subheader("Verified Vendor Master Management")

    st.write(
        "Register each legitimate vendor and their trusted bank "
        "account. Invoice analysis will compare uploaded accounts "
        "against these records."
    )

    if db is None:
        st.error(
            "Supabase is not connected. Configure the database "
            "secrets before using this page."
        )
    else:
        with st.form("vendor_form"):
            vendor_id = st.text_input(
                "Vendor ID",
                placeholder="Example: V001"
            ).strip()

            vendor_name = st.text_input(
                "Vendor Name",
                placeholder="Example: ABC Supplies"
            ).strip()

            bank_account = st.text_input(
                "Verified Bank Account",
                placeholder="Enter the trusted account identifier"
            ).strip()

            confirmed = st.checkbox(
                "I have verified these vendor details."
            )

            submitted = st.form_submit_button(
                "Register / Update Vendor"
            )

        if submitted:
            if not vendor_id or not vendor_name or not bank_account:
                st.error("Complete all three fields.")
            elif not confirmed:
                st.error(
                    "Confirm that you verified the vendor details."
                )
            else:
                try:
                    existing = (
                        db.table("vendors")
                        .select("vendor_id, verified_bank_account")
                        .eq("vendor_id", vendor_id)
                        .execute()
                    )

                    existing_rows = existing.data or []

                    if existing_rows:
                        old_account = str(
                            existing_rows[0]["verified_bank_account"]
                        ).strip()

                        if old_account != bank_account:
                            st.warning(
                                "You are changing an existing verified "
                                "bank account. Confirm that the new "
                                "account has been independently verified."
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

                    st.success(
                        f"Vendor {vendor_id} saved successfully."
                    )

                except Exception as e:
                    st.error(
                        f"Could not save vendor: {e}"
                    )

        st.divider()
        st.markdown("### Registered Vendors")

        if st.button("Refresh Vendor List"):
            st.session_state.refresh_vendors = True

        try:
            vendor_response = (
                db.table("vendors")
                .select(
                    "vendor_id, vendor_name, "
                    "verified_bank_account, registered_at, updated_at"
                )
                .order("vendor_id")
                .execute()
            )

            vendor_data = vendor_response.data or []

            if vendor_data:
                st.dataframe(
                    pd.DataFrame(vendor_data),
                    use_container_width=True
                )
            else:
                st.info(
                    "No vendors registered yet. Add verified vendor "
                    "records using the form above."
                )

        except Exception as e:
            st.error(
                f"Could not load vendor records: {e}"
            )


# =========================================================
# 11. INVESTIGATION HISTORY PAGE
# =========================================================

elif page == "Investigation History":
    st.subheader("Persistent Investigation History")

    if db is None:
        st.error(
            "Connect Supabase to view saved investigation records."
        )
    else:
        if st.button("Refresh Investigation History"):
            st.session_state.refresh_history = True

        try:
            history_response = (
                db.table("investigation_history")
                .select(
                    "id, invoice_id, vendor_id, detected_issue, "
                    "risk_score, investigated_at"
                )
                .order("investigated_at", desc=True)
                .limit(1000)
                .execute()
            )

            history = history_response.data or []

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
                    data=history_df.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="investigation_history.csv",
                    mime="text/csv"
                )

            else:
                st.info(
                    "No investigation records saved yet. Run invoice "
                    "analysis and click Save Suspicious Invoices to Database."
                )

        except Exception as e:
            st.error(
                f"Could not load investigation history: {e}"
            )


# =========================================================
# 12. METHODOLOGY PAGE
# =========================================================

elif page == "Methodology":
    st.subheader("Detection Methodology")

    st.markdown("""
    **1. Data validation:** checks required fields, valid dates,
    positive amounts and missing values.

    **2. Duplicate detection:** identifies repeated invoice IDs.

    **3. Amount anomaly detection:** compares invoice amounts with
    the typical amount for that vendor in the uploaded dataset.

    **4. Vendor master verification:** compares each invoice's bank
    account with the independently verified vendor account in Supabase.

    **5. Batch account checks:** identifies accounts that differ from
    the dominant account pattern for the vendor in the current upload.

    **6. Machine learning:** Isolation Forest identifies unusual
    records using invoice amounts and optional payment timing.

    **7. Risk scoring:** combines rule-based indicators and ML flags
    into a transparent 0–100 prioritization score.

    **8. Vendor analytics:** summarizes invoice activity and suspicious
    indicators for each vendor.

    **9. Investigation history:** stores flagged findings in Supabase
    when an authorized user explicitly saves the analysis.
    """)

    st.warning(
        "The risk score is not a calibrated fraud probability. "
        "Unregistered vendors require review, and a detected anomaly "
        "must be investigated before concluding that fraud occurred."
    )

    st.markdown("### Current limitations")

    st.markdown("""
    - Isolation Forest is trained on the uploaded dataset, so its
      findings depend on the data provided.
    - Amount thresholds are initial prototype rules and require
      validation using representative data.
    - Investigation records are saved as history entries; repeated
      saves can create multiple entries for the same invoice.
    - Vendor records must be verified and maintained by authorized staff.
    """)

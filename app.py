import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from io import StringIO

st.set_page_config(
    page_title="Invoice Fraud & Vendor Risk AI",
    page_icon="🔎",
    layout="wide"
)

REQUIRED = [
    "invoice_id", "vendor_id", "invoice_date",
    "amount", "bank_account"
]

def make_demo_data(n=600, seed=42):
    rng = np.random.default_rng(seed)
    vendors = [f"V{i:03d}" for i in range(1, 31)]
    accounts = {v: f"ACC{i:04d}" for i, v in enumerate(vendors)}

    df = pd.DataFrame({
        "invoice_id": [f"INV{i:05d}" for i in range(n)],
        "vendor_id": rng.choice(vendors, n),
        "invoice_date": (
            pd.Timestamp("2025-01-01")
            + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
        ),
        "amount": np.round(rng.lognormal(8.5, 0.8, n), 2),
        "payment_days": rng.integers(1, 61, n)
    })

    df["bank_account"] = df["vendor_id"].map(accounts)
    df["synthetic_label"] = 0

    # Controlled suspicious examples for demonstration only.
    suspicious = rng.choice(df.index, size=20, replace=False)
    df.loc[suspicious, "amount"] = rng.uniform(
        50000, 120000, len(suspicious)
    ).round(2)
    df.loc[suspicious, "synthetic_label"] = 1

    dup = rng.choice(df.index, size=8, replace=False)
    for idx in dup:
        source = max(0, idx - 1)
        df.loc[idx, "invoice_id"] = df.loc[source, "invoice_id"]
        df.loc[idx, "synthetic_label"] = 1

    changed = rng.choice(df.index, size=8, replace=False)
    df.loc[changed, "bank_account"] = [
        f"CHANGED{i}" for i in changed
    ]
    df.loc[changed, "synthetic_label"] = 1

    df["invoice_date"] = df["invoice_date"].dt.strftime("%Y-%m-%d")
    return df

def analyze_invoices(raw):
    df = raw.copy()
    missing = set(REQUIRED) - set(df.columns)

    if missing:
        raise ValueError(
            "Missing columns: " + ", ".join(sorted(missing))
        )

    for col in ["invoice_id", "vendor_id", "bank_account"]:
        df[col] = df[col].astype("string").str.strip()

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["invoice_date"] = pd.to_datetime(
        df["invoice_date"], errors="coerce"
    )

    if df[REQUIRED].isnull().any().any():
        raise ValueError(
            "Required fields contain missing or invalid values."
        )

    if (df["amount"] <= 0).any():
        raise ValueError("Invoice amounts must be positive.")

    if (df[["invoice_id", "vendor_id", "bank_account"]] == "").any().any():
        raise ValueError("Invoice ID, vendor ID and bank account cannot be blank.")

    # Rule 1: duplicate invoice IDs.
    df["duplicate_invoice"] = df.duplicated(
        subset=["invoice_id"], keep=False
    )

    # Rule 2: unusual amount compared with vendor's median.
    vendor_median = df.groupby("vendor_id")["amount"].transform("median")
    df["unusual_amount"] = (
        df["amount"] > np.maximum(vendor_median * 3, 10000)
    )

    # Rule 3: bank-account inconsistency within this uploaded batch.
    # This is not a substitute for a verified vendor master database.
    expected_account = (
        df.groupby("vendor_id")["bank_account"]
        .agg(lambda s: s.mode().iloc[0])
    )
    df["account_mismatch"] = df.apply(
        lambda r: r["bank_account"] != expected_account[r["vendor_id"]],
        axis=1
    )

    # ML anomaly detection. Uses amounts and optional payment timing.
    features = pd.DataFrame({
        "log_amount": np.log1p(df["amount"]),
    })

    if "payment_days" in df.columns:
        payment = pd.to_numeric(df["payment_days"], errors="coerce")
        features["payment_days"] = payment.fillna(payment.median())

    if len(df) >= 10 and features.nunique().sum() > 1:
        model = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=42
        )
        df["ml_anomaly"] = model.fit_predict(features) == -1
    else:
        df["ml_anomaly"] = False

    # Transparent rule-based score; not a probability of fraud.
    df["risk_score"] = (
        df["duplicate_invoice"].astype(int) * 40
        + df["unusual_amount"].astype(int) * 25
        + df["account_mismatch"].astype(int) * 30
        + df["ml_anomaly"].astype(int) * 15
    ).clip(upper=100)

    df["risk_level"] = pd.cut(
        df["risk_score"],
        bins=[-1, 24, 49, 100],
        labels=["Low", "Medium", "High"]
    ).astype(str)

    def reasons(r):
        items = []
        if r["duplicate_invoice"]:
            items.append("Duplicate invoice ID")
        if r["unusual_amount"]:
            items.append("Unusually high amount")
        if r["account_mismatch"]:
            items.append("Account differs from batch pattern")
        if r["ml_anomaly"]:
            items.append("ML anomaly")
        return "; ".join(items) if items else "No rule triggered"

    df["risk_reasons"] = df.apply(reasons, axis=1)
    return df

st.title("🔎 AI-Based Invoice Fraud & Vendor Risk Detection")
st.caption(
    "Final-year project prototype | ML anomaly detection, "
    "invoice rules and vendor analytics"
)

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose page",
        ["Overview", "Analyze Invoices", "Vendor Risk", "Methodology"]
    )
    st.divider()
    st.caption("Use synthetic or authorized data only.")

if "demo_df" not in st.session_state:
    st.session_state.demo_df = None
if "results" not in st.session_state:
    st.session_state.results = None

if page == "Overview":
    st.subheader("Project overview")
    st.write(
        "Upload invoice records to identify suspicious patterns, "
        "prioritize vendor reviews and download an analysis report."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Detection methods", "4")
    c2.metric("Input format", "CSV")
    c3.metric("Database", "Not connected yet")

    st.info(
        "Start with Analyze Invoices. Generate demonstration data or "
        "upload your own CSV."
    )

    st.markdown("""
    **Included features**
    - Duplicate invoice detection
    - Vendor-relative amount anomaly checks
    - Bank-account inconsistency checks
    - Isolation Forest anomaly detection
    - Vendor-level risk summary and CSV export
    """)

elif page == "Analyze Invoices":
    st.subheader("Upload or generate invoice data")

    if st.button("Generate sample invoices"):
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
        st.write("Input preview")
        st.dataframe(source.head(10), use_container_width=True)

        st.caption("Required columns: " + ", ".join(REQUIRED))

        if st.button("Run AI fraud analysis", type="primary"):
            try:
                st.session_state.results = analyze_invoices(source)
                st.success("Analysis completed.")
            except Exception as e:
                st.session_state.results = None
                st.error(str(e))

    results = st.session_state.results

    if results is not None:
        st.divider()
        st.subheader("Analysis results")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Invoices", len(results))
        c2.metric("High risk", int((results["risk_level"] == "High").sum()))
        c3.metric("Medium risk", int((results["risk_level"] == "Medium").sum()))
        c4.metric("Vendors", results["vendor_id"].nunique())

        st.dataframe(
            results.sort_values("risk_score", ascending=False),
            use_container_width=True
        )

        st.download_button(
            "Download full analysis CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="invoice_fraud_analysis.csv",
            mime="text/csv"
        )

        st.warning(
            "A high-risk flag is not proof of fraud. Review the evidence "
            "before making payment or vendor decisions."
        )

elif page == "Vendor Risk":
    st.subheader("Vendor risk analytics")
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
                "risk_level", lambda x: (x == "High").sum()
            ),
            account_mismatches=("account_mismatch", "sum")
        ).reset_index()

        vendors = vendors.sort_values(
            ["high_risk_invoices", "maximum_risk_score"],
            ascending=False
        )

        st.dataframe(vendors, use_container_width=True)

        fig = px.bar(
            vendors.head(15),
            x="vendor_id",
            y="total_invoice_value",
            title="Invoice value by vendor"
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(
            vendors,
            x="invoice_count",
            y="maximum_risk_score",
            size="total_invoice_value",
            hover_name="vendor_id",
            title="Vendor activity versus maximum risk score"
        )
        st.plotly_chart(fig2, use_container_width=True)

elif page == "Methodology":
    st.subheader("Detection methodology")

    st.markdown("""
    **1. Data validation:** required columns, dates and positive amounts.

    **2. Rule-based detection:** duplicate invoice IDs, unusually large
    amounts and bank-account inconsistencies within the uploaded batch.

    **3. Unsupervised machine learning:** Isolation Forest flags unusual
    observations based on the amount and, if available, payment timing.

    **4. Risk scoring:** rule indicators and ML flags are combined into a
    transparent 0–100 prioritization score.

    **5. Vendor analytics:** aggregate invoice activity and suspicious
    indicators by vendor.
    """)

    st.warning(
        "The score is a prioritization score, not a calibrated fraud "
        "probability. Batch-based account checks must be replaced with "
        "verified vendor master records for a production-quality system."
    )

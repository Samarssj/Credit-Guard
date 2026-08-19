from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "fraud_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
METADATA_PATH = MODEL_DIR / "metadata.json"
PREDICTIONS_PATH = MODEL_DIR / "test_predictions.csv"
COMPARISON_PATH = MODEL_DIR / "model_comparison.csv"
FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]

st.set_page_config(
    page_title="CreditGuard Risk Console",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        :root {
            --ink: #dce9f8;
            --muted: #8da4bf;
            --navy: #06111f;
            --panel: rgba(13, 29, 49, 0.78);
            --line: rgba(126, 184, 153, 0.18);
            --cyan: #57e39b;
            --blue: #2f9e67;
            --red: #ff6b7d;
            --green: #45e0ae;
        }
        html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
        .stApp {
            background:
                radial-gradient(circle at 8% 0%, rgba(44, 107, 178, 0.22), transparent 28rem),
                radial-gradient(circle at 88% 12%, rgba(0, 214, 255, 0.11), transparent 24rem),
                linear-gradient(135deg, #06111f 0%, #07182a 46%, #091d32 100%);
            color: var(--ink);
        }
        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            opacity: .21;
            background-image: linear-gradient(rgba(110, 163, 207, .08) 1px, transparent 1px), linear-gradient(90deg, rgba(110, 163, 207, .08) 1px, transparent 1px);
            background-size: 46px 46px;
            mask-image: linear-gradient(to bottom, black, transparent 80%);
        }
        header[data-testid="stHeader"] { background: transparent !important; height: 0 !important; }
        [data-testid="stToolbar"] { display: none !important; }
        [data-testid="stSidebar"] { display: none !important; }
        .block-container { max-width: 100%; padding: .65rem 1.25rem 3rem; }
        .top-control-bar { display: flex; align-items: center; gap: .8rem; margin: .2rem 0 1rem; padding: .55rem .7rem; border: 1px solid rgba(126, 184, 153, .18); border-radius: 16px; background: rgba(7, 22, 38, .82); box-shadow: 0 14px 35px rgba(0,0,0,.16); }
        .top-control-label { color: var(--cyan); font-size: .64rem; font-weight: 850; letter-spacing: .14em; text-transform: uppercase; margin: .1rem 0 .18rem; }
        .top-control-caption { color: #8da4bf; font-size: .72rem; line-height: 1.25; }
        .top-status { padding: .55rem .75rem; border: 1px solid rgba(69,224,174,.25); border-radius: 11px; background: rgba(69,224,174,.07); color: #8cf3d1; font-size: .72rem; font-weight: 800; }
        .top-nav [role="radiogroup"] { gap: .35rem; }
        .top-nav [role="radio"] { padding: .44rem .7rem; border-radius: 9px; color: #b9cde1; }
        .top-nav [role="radio"]:hover { background: rgba(87,227,155,.08); }
        .card-symbol { position: relative; width: 36px; height: 26px; flex: 0 0 auto; display: inline-block; border-radius: 6px; background: linear-gradient(135deg, #6be5ff, #4f80ff); box-shadow: 0 0 25px rgba(87,227,155,.3); }
        .card-symbol::before { content: ""; position: absolute; left: 6px; top: 7px; width: 8px; height: 6px; border-radius: 2px; background: #f7d889; box-shadow: inset 0 0 0 1px rgba(56,46,16,.35); }
        .card-symbol::after { content: ""; position: absolute; left: 0; right: 0; top: 13px; height: 2px; background: rgba(3,17,30,.36); }
        [data-testid="stSlider"] { padding: 0; }
        [data-testid="stSlider"] [role="slider"] { background: var(--cyan); border-color: white; }
        .top-brand { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.15rem; }
        .brand-lockup { display: flex; align-items: center; gap: .7rem; }
        .brand-mark { width: 36px; height: 36px; display: grid; place-items: center; border-radius: 11px; color: #03111e; background: linear-gradient(135deg, var(--cyan), #9cf4ff); box-shadow: 0 0 28px rgba(87,227,155,.34); font-size: 1.25rem; }
        .brand-name { font-size: 1.08rem; letter-spacing: .02em; font-weight: 800; color: white; }
        .brand-sub { color: var(--muted); font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; }
        .live-pill, .secure-pill { border: 1px solid rgba(69,224,174,.35); background: rgba(69,224,174,.09); color: #84f2cd; border-radius: 99px; padding: .42rem .75rem; font-size: .72rem; font-weight: 800; letter-spacing: .09em; text-transform: uppercase; }
        .hero-shell { position: relative; overflow: hidden; min-height: 270px; border: 1px solid rgba(127, 192, 158, .22); border-radius: 24px; padding: 2rem 2.2rem; background: linear-gradient(110deg, rgba(11, 35, 60, .95), rgba(8, 23, 41, .82)); box-shadow: 0 22px 70px rgba(0,0,0,.24), inset 0 1px 0 rgba(255,255,255,.05); }
        .hero-shell::after { content: ""; position: absolute; width: 370px; height: 370px; right: -120px; top: -145px; border-radius: 50%; background: radial-gradient(circle, rgba(87,227,155,.22), transparent 64%); }
        .hero-copy { position: relative; z-index: 2; max-width: 53%; }
        .eyebrow { color: var(--cyan); font-size: .72rem; font-weight: 800; letter-spacing: .18em; text-transform: uppercase; margin-bottom: .7rem; }
        .hero-title { color: white; font-size: clamp(2.1rem, 4vw, 3.8rem); line-height: .98; letter-spacing: -.065em; margin: 0; font-weight: 850; }
        .hero-text { color: #a9bfd4; font-size: 1rem; line-height: 1.55; margin-top: 1rem; max-width: 570px; }
        .credit-card-scene { position: absolute; z-index: 1; right: 5%; top: 28px; width: 340px; height: 214px; transform: rotate(-8deg); filter: drop-shadow(0 24px 25px rgba(0,0,0,.36)); }
        .credit-card { position: absolute; inset: 0; border-radius: 20px; padding: 1.4rem 1.45rem; overflow: hidden; background: linear-gradient(135deg, #123b33 0%, #187358 43%, #15513d 100%); border: 1px solid rgba(190, 239, 255, .48); }
        .credit-card::before { content: ""; position: absolute; width: 270px; height: 270px; right: -105px; top: -125px; border: 1px solid rgba(255,255,255,.2); border-radius: 50%; box-shadow: -18px 0 0 rgba(255,255,255,.08), -36px 0 0 rgba(255,255,255,.05); }
        .credit-card::after { content: ""; position: absolute; inset: 0; background: linear-gradient(115deg, transparent 18%, rgba(255,255,255,.12) 38%, transparent 58%); transform: translateX(-100%); animation: sheen 7s infinite; }
        @keyframes sheen { 0%, 50% { transform: translateX(-100%); } 72%, 100% { transform: translateX(100%); } }
        .card-top, .card-bottom { position: relative; z-index: 2; display: flex; justify-content: space-between; align-items: center; }
        .card-network { color: white; font-weight: 850; font-size: 1.08rem; letter-spacing: .08em; }
        .card-network span { color: var(--cyan); }
        .card-chip { width: 43px; height: 31px; border-radius: 8px; background: linear-gradient(135deg, #f4d987, #b98932); border: 1px solid rgba(255,255,255,.55); position: relative; }
        .card-chip::after { content: ""; position: absolute; inset: 6px 0; border-top: 1px solid rgba(60,40,12,.4); border-bottom: 1px solid rgba(60,40,12,.4); }
        .card-number { position: absolute; z-index: 2; left: 1.45rem; bottom: 4.7rem; color: #eaf8ff; font-size: 1rem; letter-spacing: .22em; text-shadow: 0 2px 8px rgba(0,0,0,.3); }
        .card-bottom { position: absolute; left: 1.45rem; right: 1.45rem; bottom: 1.25rem; color: #c3e1ee; font-size: .58rem; letter-spacing: .1em; text-transform: uppercase; }
        .threshold-banner { display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin: 1rem 0 1.4rem; padding: .86rem 1.1rem; border-radius: 15px; border: 1px solid rgba(87,227,155,.23); background: linear-gradient(90deg, rgba(19, 77, 54, .72), rgba(13, 32, 55, .7)); box-shadow: inset 0 1px 0 rgba(255,255,255,.04); }
        .threshold-label { color: #b9cede; font-size: .78rem; text-transform: uppercase; letter-spacing: .11em; font-weight: 800; }
        .threshold-value { color: white; font-size: 1.22rem; font-weight: 850; }
        .threshold-copy { color: #91abc3; font-size: .85rem; text-align: right; }
        .section-kicker { color: var(--cyan); letter-spacing: .15em; text-transform: uppercase; font-weight: 800; font-size: .72rem; margin-bottom: .45rem; }
        h1, h2, h3, h4, p, label, .stMarkdown { color: var(--ink); }
        h2 { letter-spacing: -.035em; }
        .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 1.15rem 1.25rem; box-shadow: 0 16px 40px rgba(0,0,0,.13); }
        .panel-title { color: white; font-size: 1.08rem; font-weight: 800; margin-bottom: .35rem; }
        .panel-copy { color: #95abc2; font-size: .9rem; line-height: 1.55; }
        div[data-testid="stMetric"] { background: linear-gradient(145deg, rgba(17, 58, 42, .92), rgba(10, 25, 43, .92)); border: 1px solid rgba(126, 184, 153, .18); border-radius: 15px; padding: .95rem 1rem; box-shadow: 0 11px 28px rgba(0,0,0,.13); }
        div[data-testid="stMetric"] label { color: #8da4bf !important; font-size: .72rem; text-transform: uppercase; letter-spacing: .09em; }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: white; font-weight: 800; }
        .stButton > button, .stDownloadButton > button, button[kind="primary"] { border-radius: 11px; border: 1px solid rgba(114, 230, 176, .38); color: #041321; background: linear-gradient(135deg, #72e6b0, #2f9e67); font-weight: 850; box-shadow: 0 10px 24px rgba(35, 158, 102, .2); }
        .stButton > button:hover, .stDownloadButton > button:hover { border-color: white; color: #041321; }
        input, textarea, [data-baseweb="select"] > div { background: rgba(5, 18, 32, .76) !important; color: white !important; border-color: rgba(126, 184, 153, .25) !important; }
        [data-testid="stFileUploader"] { background: rgba(5, 18, 32, .45); border: 1px dashed rgba(89, 211, 146, .35); border-radius: 15px; padding: .4rem; }
        [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
        .footer-note { color: #718aa3; font-size: .78rem; text-align: center; padding-top: .7rem; }
        @media (max-width: 980px) { .credit-card-scene { opacity: .34; right: -5%; } .hero-copy { max-width: 85%; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_evaluation_data() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(PREDICTIONS_PATH)


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame:
    if not COMPARISON_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(COMPARISON_PATH)


metadata = load_json(METADATA_PATH)
metrics = load_json(METRICS_PATH)
model = load_model()
evaluation_df = load_evaluation_data()
comparison_df = load_model_comparison()
threshold = float(metadata.get("threshold_default", 0.50))

st.markdown("<div class='top-control-bar'>", unsafe_allow_html=True)
top_nav, top_threshold, top_status = st.columns([1.55, 1.0, .78])
with top_nav:
    st.markdown("<div class='top-control-label'>Workspace</div>", unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["Overview", "Single transaction", "Batch scoring", "Model evaluation"],
        horizontal=True,
        label_visibility="collapsed",
    )
with top_threshold:
    st.markdown("<div class='top-control-label'>Decision threshold</div>", unsafe_allow_html=True)
    threshold = st.slider(
        "Flag as fraud at or above",
        min_value=0.05,
        max_value=0.95,
        value=float(metadata.get("threshold_default", 0.50)),
        step=0.01,
        format="%.2f",
        label_visibility="collapsed",
    )
with top_status:
    st.markdown("<div class='top-control-label'>System status</div>", unsafe_allow_html=True)
    if model is None:
        st.markdown("<div class='top-status' style='border-color:rgba(255,107,125,.35);background:rgba(255,107,125,.08);color:#ff9aa8'>● Artifact unavailable</div>", unsafe_allow_html=True)
        st.caption("Run train_model.py")
    else:
        st.markdown("<div class='top-status'>● Inference engine online</div>", unsafe_allow_html=True)
        st.caption(str(metadata.get("model_name", "Fraud model")))
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="top-brand">
        <div class="brand-lockup">
            <span class="card-symbol" aria-hidden="true"></span>
            <div><div class="brand-name">CREDITGUARD / RISK CONSOLE</div><div class="brand-sub">Transaction intelligence platform</div></div>
        </div>
        <div class="live-pill">● Live scoring environment</div>
    </div>
    <div class="hero-shell">
        <div class="hero-copy">
            <div class="eyebrow">Payment security / real-time decision support</div>
            <h1 class="hero-title">Trust every<br/>transaction.</h1>
            <p class="hero-text">A focused fraud-screening workspace for identifying anomalous card activity, understanding model confidence, and routing high-risk transactions for review.</p>
        </div>
        <div class="credit-card-scene" aria-hidden="true">
            <div class="credit-card">
                <div class="card-top"><div class="card-network">CREDIT<span>GUARD</span></div><div class="card-chip"></div></div>
                <div class="card-number">••••  ••••  ••••  4826</div>
                <div class="card-bottom"><span>SECURE PAYMENT NETWORK</span><span>VALID / 08·29</span></div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="threshold-banner">
        <div><div class="threshold-label">Decision policy</div><div class="threshold-value">Review at {threshold:.0%} fraud probability</div></div>
        <div class="threshold-copy">Fixed operating threshold · Lower scores pass as likely legitimate<br/>Higher scores route to review / likely fraud</div>
    </div>
    """,
    unsafe_allow_html=True,
)


def metric_or_dash(key: str) -> str:
    value = metrics.get(key)
    return "—" if value is None else f"{float(value):.3f}"


def score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if model is None:
        raise RuntimeError("The model artifact is unavailable.")
    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")
    features = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        bad = features.columns[features.isna().any()].tolist()
        raise ValueError(f"Non-numeric or missing values found in: {', '.join(bad)}")
    result = frame.copy()
    probability = model.predict_proba(features)[:, 1]
    result["fraud_probability"] = probability
    result["decision"] = np.where(probability >= threshold, "Review / likely fraud", "Likely legitimate")
    return result


def show_overview() -> None:
    if not metadata:
        st.warning("No trained model metadata is available yet. Run the training script first.")
        return
    st.markdown("<div class='section-kicker'>Command center</div>", unsafe_allow_html=True)
    st.header("Portfolio risk overview")
    st.markdown("<div class='panel-copy'>A compact view of the benchmark population, model quality, and current decision policy.</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)

    cols = st.columns(5)
    cols[0].metric("Transactions", f"{int(metadata.get('dataset_rows', 0)):,}")
    cols[1].metric("Fraud cases", f"{int(metadata.get('fraud_count', 0)):,}")
    cols[2].metric("Fraud rate", f"{float(metadata.get('fraud_rate', 0)) * 100:.3f}%")
    cols[3].metric("Average precision", metric_or_dash("average_precision"))
    cols[4].metric("ROC-AUC", metric_or_dash("roc_auc"))

    st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.07, .93])
    with left:
        st.markdown("<div class='panel'><div class='panel-title'>Model selection ledger</div><div class='panel-copy'>The saved inference engine is selected by cross-validated average precision and refit on the full training split.</div></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:.65rem'></div>", unsafe_allow_html=True)
        if not comparison_df.empty:
            display_comparison = comparison_df.rename(columns={"model": "Model", "mean_average_precision": "Mean average precision", "mean_roc_auc": "Mean ROC-AUC", "rank_average_precision": "AP rank"})
            st.dataframe(display_comparison[["Model", "Mean average precision", "Mean ROC-AUC", "AP rank"]].style.format({"Mean average precision": "{:.3f}", "Mean ROC-AUC": "{:.3f}"}), hide_index=True, use_container_width=True)
        config = pd.DataFrame({"Setting": ["Selected model", "Resampling", "Feature treatment", "Decision threshold", "Held-out rows"], "Value": [metadata.get("model_name", "—"), f"SMOTE ratio {metadata.get('smote_sampling_strategy', '—')}", "Robust scale Time + Amount", f"{threshold:.0%}", f"{int(metadata.get('test_rows', 0)):,}"]})
        st.dataframe(config, hide_index=True, use_container_width=True)
    with right:
        counts = pd.DataFrame({"Class": ["Legitimate", "Fraud"], "Transactions": [metadata.get("legitimate_count", 0), metadata.get("fraud_count", 0)]})
        fig = px.bar(counts, x="Class", y="Transactions", color="Class", color_discrete_map={"Legitimate": "#57e39b", "Fraud": "#ff6b7d"}, title="Population balance")
        fig.update_layout(showlegend=False, height=340, margin=dict(l=10, r=10, t=55, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#dce9f8"}, xaxis={"gridcolor": "rgba(126,184,153,.12)"}, yaxis={"gridcolor": "rgba(126,184,153,.12)"})
        st.plotly_chart(fig, use_container_width=True)


def show_single_transaction() -> None:
    st.markdown("<div class='section-kicker'>Decision workspace</div>", unsafe_allow_html=True)
    st.header("Screen a transaction")
    st.markdown("<div class='panel-copy'>Enter the 30 numeric fields expected by the model. The decision policy is fixed at the threshold shown above.</div>", unsafe_allow_html=True)
    if model is None:
        st.error("The model artifact is unavailable. Train the model locally and place `models/fraud_pipeline.joblib` in this project.")
        return
    with st.form("single_transaction_form"):
        first = st.columns(2)
        with first[0]:
            time_value = st.number_input("Time", min_value=0.0, value=0.0, step=1.0)
        with first[1]:
            amount_value = st.number_input("Amount", min_value=0.0, value=100.0, step=1.0)
        st.markdown("##### Anonymized PCA feature vector")
        feature_cols = st.columns(4)
        values: dict[str, float] = {}
        for i in range(1, 29):
            with feature_cols[(i - 1) % 4]:
                values[f"V{i}"] = st.number_input(f"V{i}", value=0.0, step=0.01, format="%.6f")
        submitted = st.form_submit_button("Run secure screening", type="primary", use_container_width=True)
    if submitted:
        row = {"Time": time_value, **values, "Amount": amount_value}
        scored = score_frame(pd.DataFrame([row]))
        probability = float(scored.loc[0, "fraud_probability"])
        decision = scored.loc[0, "decision"]
        st.divider()
        if probability >= threshold:
            st.error(f"Decision: {decision}")
        else:
            st.success(f"Decision: {decision}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Fraud probability", f"{probability:.2%}")
        m2.metric("Policy threshold", f"{threshold:.0%}")
        m3.metric("Model score", f"{probability:.4f}")
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=probability * 100, number={"suffix": "%"}, title={"text": "Fraud probability"}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#ff6b7d" if probability >= threshold else "#57e39b"}, "threshold": {"line": {"color": "#f4f8ff", "width": 4}, "thickness": 0.8, "value": threshold * 100}}))
        gauge.update_layout(height=290, margin=dict(l=35, r=35, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", font={"color": "#dce9f8"})
        st.plotly_chart(gauge, use_container_width=True)
        st.caption("Benchmark-model score for decision support; use human review for operational decisions.")


def show_batch_scoring() -> None:
    st.markdown("<div class='section-kicker'>Operations</div>", unsafe_allow_html=True)
    st.header("Score a transaction file")
    st.markdown("<div class='panel-copy'>Upload a CSV containing `Time`, `V1`–`V28`, and `Amount`. The fixed decision policy will be applied to every row.</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.info("No file uploaded. Use the transaction screen for a single decision or upload a scored batch here.")
        return
    try:
        frame = pd.read_csv(uploaded)
        scored = score_frame(frame)
    except Exception as exc:
        st.error(str(exc))
        return
    fraud_count = int((scored["fraud_probability"] >= threshold).sum())
    st.success(f"Scored {len(scored):,} rows; {fraud_count:,} crossed the review threshold.")
    c1, c2 = st.columns(2)
    c1.metric("Review queue", f"{fraud_count:,}")
    c2.metric("Likely legitimate", f"{len(scored) - fraud_count:,}")
    st.dataframe(scored[[*FEATURE_COLUMNS[:2], "Amount", "fraud_probability", "decision"] + (["Class"] if "Class" in scored else [])], use_container_width=True, height=420)
    st.download_button("Download scored CSV", scored.to_csv(index=False).encode("utf-8"), "fraud_scored_transactions.csv", "text/csv", use_container_width=True)


def show_evaluation() -> None:
    st.markdown("<div class='section-kicker'>Model observability</div>", unsafe_allow_html=True)
    st.header("Model evaluation")
    if not metrics or evaluation_df.empty:
        st.warning("Evaluation artifacts are unavailable. Run the training script first.")
        return
    st.markdown("<div class='panel-copy'>Metrics below are calculated on the untouched stratified test split. The operating threshold is fixed at the policy value displayed above.</div>", unsafe_allow_html=True)
    y_true = evaluation_df["actual_class"].astype(int).to_numpy()
    probabilities = evaluation_df["fraud_probability"].to_numpy()
    predictions = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    cols = st.columns(5)
    cols[0].metric("Precision", f"{((cm[1,1] / (cm[1,1] + cm[0,1])) if (cm[1,1] + cm[0,1]) else 0):.3f}")
    cols[1].metric("Recall", f"{((cm[1,1] / (cm[1,1] + cm[1,0])) if (cm[1,1] + cm[1,0]) else 0):.3f}")
    p = (cm[1,1] / (cm[1,1] + cm[0,1])) if (cm[1,1] + cm[0,1]) else 0
    r = (cm[1,1] / (cm[1,1] + cm[1,0])) if (cm[1,1] + cm[1,0]) else 0
    cols[2].metric("F1", f"{(2*p*r/(p+r) if p+r else 0):.3f}")
    cols[3].metric("Average precision", metric_or_dash("average_precision"))
    cols[4].metric("Review queue", f"{int(predictions.sum()):,}")
    left, right = st.columns(2)
    with left:
        cm_df = pd.DataFrame(cm, index=["Actual legitimate", "Actual fraud"], columns=["Predicted legitimate", "Predicted fraud"])
        fig = px.imshow(cm_df, text_auto=True, color_continuous_scale=[[0, "#0b3024"], [1, "#57e39b"]], title=f"Confusion matrix · policy {threshold:.0%}")
        fig.update_layout(height=390, margin=dict(l=10, r=10, t=55, b=10), paper_bgcolor="rgba(0,0,0,0)", font={"color": "#dce9f8"})
        st.plotly_chart(fig, use_container_width=True)
    with right:
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC curve", line={"color": "#57e39b", "width": 3}))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random baseline", line={"dash": "dash", "color": "#6d8299"}))
        roc_fig.update_layout(title=f"ROC curve · AUC {float(metrics.get('roc_auc', 0)):.3f}", xaxis_title="False-positive rate", yaxis_title="True-positive rate", height=390, margin=dict(l=10, r=10, t=55, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#dce9f8"})
        st.plotly_chart(roc_fig, use_container_width=True)
    pr_fig = go.Figure(go.Scatter(x=recall, y=precision, mode="lines", line={"color": "#ff6b7d", "width": 3}))
    pr_fig.update_layout(title=f"Precision-recall curve · AP {float(metrics.get('average_precision', 0)):.3f}", xaxis_title="Recall", yaxis_title="Precision", height=390, margin=dict(l=10, r=10, t=55, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#dce9f8"})
    st.plotly_chart(pr_fig, use_container_width=True)
    st.markdown("#### Fixed policy trade-off reference")
    threshold_rows = []
    for value in np.linspace(0.05, 0.95, 19):
        pred = (probabilities >= value).astype(int)
        matrix = confusion_matrix(y_true, pred, labels=[0, 1])
        tp, fp, fn = matrix[1, 1], matrix[0, 1], matrix[1, 0]
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        threshold_rows.append({"threshold": round(float(value), 2), "precision": p, "recall": r, "flagged": int(pred.sum())})
    threshold_df = pd.DataFrame(threshold_rows)
    st.dataframe(threshold_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}"}), use_container_width=True, hide_index=True)


if page == "Overview":
    show_overview()
elif page == "Single transaction":
    show_single_transaction()
elif page == "Batch scoring":
    show_batch_scoring()
else:
    show_evaluation()

st.divider()
st.markdown("<div class='footer-note'>CREDITGUARD · Protected scoring workspace · Anonymized benchmark features · Human review recommended for flagged transactions</div>", unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="BNPL Default Risk Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

sns.set_theme(style="whitegrid", palette="muted")

RISK_COLORS = {"Low Risk": "#2ecc71", "Medium Risk": "#f39c12", "High Risk": "#e74c3c"}
CAT_COLS = ["Education", "EmploymentType", "MaritalStatus",
            "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner"]

# ─────────────────────────────────────────────
# Load data + train everything — cached once
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Loading data & training models — please wait…")
def load_and_train():
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import KMeans
    from imblearn.over_sampling import SMOTE

    # ── Raw data ────────────────────────────────────────────────────
    df = pd.read_csv("bnpl_sample_40000.csv")

    # ── Encode ──────────────────────────────────────────────────────
    df2 = df.copy()
    df2.drop("LoanID", axis=1, inplace=True)

    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        df2[col] = le.fit_transform(df2[col])
        encoders[col] = le

    X = df2.drop("Default", axis=1)
    y = df2["Default"]
    feature_names = X.columns.tolist()

    # ── Scale ───────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── SMOTE + split ───────────────────────────────────────────────
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_scaled, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

    # ── Random Forest ───────────────────────────────────────────────
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # ── KMeans ──────────────────────────────────────────────────────
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df2["Cluster"] = kmeans.fit_predict(X_scaled)

    cluster_summary = df2.groupby("Cluster")[
        ["CreditScore", "DTIRatio", "Income", "InterestRate", "Default"]
    ].mean().round(3)

    risk_order = cluster_summary["Default"].rank().astype(int).to_dict()
    label_map  = {k: ["Low Risk", "Medium Risk", "High Risk"][v - 1] for k, v in risk_order.items()}
    df2["RiskSegment"] = df2["Cluster"].map(label_map)

    # ── TOPSIS ──────────────────────────────────────────────────────
    topsis_feats = ["CreditScore", "DTIRatio", "InterestRate", "Income"]
    matrix   = df2[topsis_feats].values.astype(float)
    norm     = matrix / np.sqrt((matrix ** 2).sum(axis=0))
    weights  = np.array([0.25, 0.25, 0.25, 0.25])
    weighted = norm * weights
    benefit  = [True, False, False, True]
    ib = np.where(benefit, weighted.max(0), weighted.min(0))
    iw = np.where(benefit, weighted.min(0), weighted.max(0))
    db = np.sqrt(((weighted - ib) ** 2).sum(axis=1))
    dw = np.sqrt(((weighted - iw) ** 2).sum(axis=1))
    df2["TOPSIS_Score"] = dw / (db + dw)

    return (df, df2, model, scaler, kmeans,
            label_map, feature_names, encoders, cluster_summary)

(df, df2, model, scaler, kmeans,
 label_map, feature_names, encoders, cluster_summary) = load_and_train()

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.title("🏦 BNPL Risk Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview & EDA", "🔍 Predict Risk", "📌 Customer Segments", "📐 TOPSIS Scoring"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Dataset : 40,000 BNPL Loan Records")
st.sidebar.caption("Model   : Random Forest + SMOTE")
st.sidebar.caption("Cluster : KMeans (k=3)")

# ═══════════════════════════════════════════════════════
# PAGE 1 — Overview & EDA
# ═══════════════════════════════════════════════════════
if page == "📊 Overview & EDA":
    st.title("📊 BNPL Default Risk — Exploratory Data Analysis")
    st.markdown("---")

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",    f"{len(df):,}")
    c2.metric("Default Rate",     f"{df['Default'].mean()*100:.2f}%")
    c3.metric("Avg Credit Score", f"{df['CreditScore'].mean():.0f}")
    c4.metric("Avg DTI Ratio",    f"{df['DTIRatio'].mean():.3f}")

    st.markdown("---")

    # Default distribution
    st.subheader("📌 Default Distribution")
    counts = df["Default"].value_counts()
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(["Safe (0)", "Risky (1)"], counts.values,
                      color=["#2ecc71", "#e74c3c"], edgecolor="black")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                    f"{int(bar.get_height()):,}", ha="center", fontsize=11, fontweight="bold")
        ax.set_ylabel("Count")
        ax.set_title("Count by Class", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(counts.values, labels=["Safe", "Risky"], autopct="%1.1f%%",
               colors=["#2ecc71", "#e74c3c"], startangle=90,
               wedgeprops=dict(edgecolor="white"))
        ax.set_title("Proportion", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")

    # Numerical distributions
    st.subheader("📉 Numerical Feature Distributions by Default Status")
    num_cols = ["CreditScore", "DTIRatio", "Income", "LoanAmount", "InterestRate", "Age"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.histplot(data=df, x=col, hue="Default", kde=True, ax=axes[i],
                     palette={0: "#2ecc71", 1: "#e74c3c"}, alpha=0.6, bins=40)
        axes[i].set_title(col, fontweight="bold")
        axes[i].legend(["Safe", "Risky"], title="Default")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")

    # Categorical default rates
    st.subheader("📋 Default Rate by Categorical Feature")
    selected_cat = st.selectbox("Select Feature", CAT_COLS)
    fig, ax = plt.subplots(figsize=(8, 4))
    dr = df.groupby(selected_cat)["Default"].mean() * 100
    dr.sort_values().plot(kind="bar", ax=ax, color="#e74c3c", edgecolor="black", alpha=0.8)
    ax.set_title(f"Default Rate by {selected_cat}", fontweight="bold")
    ax.set_ylabel("Default Rate (%)")
    ax.tick_params(axis="x", rotation=30)
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{bar.get_height():.1f}%", ha="center", fontsize=9)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")

    # Correlation heatmap
    st.subheader("📌 Correlation Heatmap")
    numeric_df = df.select_dtypes(include="number").drop(columns=["LoanID"], errors="ignore")
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                mask=mask, linewidths=0.5, vmin=-1, vmax=1,
                annot_kws={"size": 8}, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")

    # Feature importance
    st.subheader("🌲 Feature Importance — Random Forest")
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    importances.plot(kind="barh", ax=ax, color="#3498db", edgecolor="black")
    ax.set_title("Feature Importances", fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    with st.expander("🔎 View Raw Data Sample"):
        st.dataframe(df.head(200), use_container_width=True)

# ═══════════════════════════════════════════════════════
# PAGE 2 — Predict Risk
# ═══════════════════════════════════════════════════════
elif page == "🔍 Predict Risk":
    st.title("🔍 Predict Customer Default Risk")
    st.markdown("Fill in the customer details below to get a real-time risk prediction.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("💰 Financial")
        credit_score     = st.slider("Credit Score", 300, 900, 650)
        income           = st.number_input("Annual Income", min_value=1000,
                                           max_value=500000, value=50000, step=1000)
        loan_amount      = st.number_input("Loan Amount", min_value=1000,
                                           max_value=200000, value=20000, step=1000)
        interest_rate    = st.slider("Interest Rate (%)", 0.0, 30.0, 10.0, 0.1)
        dti_ratio        = st.slider("DTI Ratio", 0.0, 1.0, 0.3, 0.01)

    with col2:
        st.subheader("👤 Personal")
        age              = st.slider("Age", 18, 75, 35)
        education        = st.selectbox("Education",      encoders["Education"].classes_)
        marital_status   = st.selectbox("Marital Status", encoders["MaritalStatus"].classes_)
        has_mortgage     = st.selectbox("Has Mortgage",   encoders["HasMortgage"].classes_)
        has_dependents   = st.selectbox("Has Dependents", encoders["HasDependents"].classes_)

    with col3:
        st.subheader("💼 Employment & Loan")
        employment_type  = st.selectbox("Employment Type", encoders["EmploymentType"].classes_)
        months_employed  = st.slider("Months Employed", 0, 120, 36)
        num_credit_lines = st.slider("Num Credit Lines", 1, 10, 3)
        loan_term        = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
        loan_purpose     = st.selectbox("Loan Purpose",  encoders["LoanPurpose"].classes_)
        has_cosigner     = st.selectbox("Has Co-Signer", encoders["HasCoSigner"].classes_)

    st.markdown("---")

    if st.button("🚀 Predict Risk", use_container_width=True):
        input_dict = {
            "Age":             age,
            "Income":          income,
            "LoanAmount":      loan_amount,
            "CreditScore":     credit_score,
            "MonthsEmployed":  months_employed,
            "NumCreditLines":  num_credit_lines,
            "InterestRate":    interest_rate,
            "LoanTerm":        loan_term,
            "DTIRatio":        dti_ratio,
            "Education":       encoders["Education"].transform([education])[0],
            "EmploymentType":  encoders["EmploymentType"].transform([employment_type])[0],
            "MaritalStatus":   encoders["MaritalStatus"].transform([marital_status])[0],
            "HasMortgage":     encoders["HasMortgage"].transform([has_mortgage])[0],
            "HasDependents":   encoders["HasDependents"].transform([has_dependents])[0],
            "LoanPurpose":     encoders["LoanPurpose"].transform([loan_purpose])[0],
            "HasCoSigner":     encoders["HasCoSigner"].transform([has_cosigner])[0],
        }

        input_df     = pd.DataFrame([input_dict])[feature_names]
        input_scaled = scaler.transform(input_df)

        pred     = model.predict(input_scaled)[0]
        prob     = model.predict_proba(input_scaled)[0][1]
        cluster  = kmeans.predict(input_scaled)[0]
        risk_seg = label_map[cluster]

        r1, r2, r3 = st.columns(3)
        with r1:
            if pred == 1:
                st.error("⚠️ **RISKY — Likely to Default**")
            else:
                st.success("✅ **SAFE — Unlikely to Default**")
        with r2:
            st.metric("Default Probability", f"{prob * 100:.1f}%")
        with r3:
            color = RISK_COLORS.get(risk_seg, "gray")
            st.markdown(
                f"**Risk Segment:** "
                f"<span style='color:{color};font-size:18px;font-weight:bold'>{risk_seg}</span>",
                unsafe_allow_html=True)

        # Probability bar
        fig, ax = plt.subplots(figsize=(7, 1.2))
        bar_color = "#e74c3c" if prob > 0.5 else "#2ecc71"
        ax.barh([""], [prob], color=bar_color, height=0.5)
        ax.barh([""], [1 - prob], left=[prob], color="#ecf0f1", height=0.5)
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.text(prob / 2, 0, f"{prob * 100:.1f}%", ha="center", va="center",
                color="white", fontweight="bold", fontsize=13)
        ax.set_title("Default Probability", fontweight="bold")
        ax.axis("off")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # TOPSIS for this individual
        ref      = df2[["CreditScore", "DTIRatio", "InterestRate", "Income"]].values.astype(float)
        new_row  = np.array([[credit_score, dti_ratio, interest_rate, income]], dtype=float)
        aug      = np.vstack([ref, new_row])
        norm_aug = aug / np.sqrt((aug ** 2).sum(axis=0))
        w        = np.array([0.25, 0.25, 0.25, 0.25])
        wt       = norm_aug * w
        benefit  = [True, False, False, True]
        ib = np.where(benefit, wt.max(0), wt.min(0))
        iw = np.where(benefit, wt.min(0), wt.max(0))
        db_val = np.sqrt(((wt[-1] - ib) ** 2).sum())
        dw_val = np.sqrt(((wt[-1] - iw) ** 2).sum())
        t_score = dw_val / (db_val + dw_val)
        st.info(f"📐 **TOPSIS Risk Score:** {t_score:.4f}  *(1 = safest, 0 = riskiest)*")

# ═══════════════════════════════════════════════════════
# PAGE 3 — Customer Segments
# ═══════════════════════════════════════════════════════
elif page == "📌 Customer Segments":
    st.title("📌 KMeans Customer Segmentation")
    st.markdown("Customers are grouped into **3 risk clusters** using KMeans on scaled features.")
    st.markdown("---")

    st.subheader("Cluster Summary")
    st.dataframe(
        cluster_summary.style.background_gradient(cmap="RdYlGn_r", subset=["Default"]),
        use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Segment Distribution")
        seg_counts = df2["RiskSegment"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(seg_counts.index, seg_counts.values,
                      color=[RISK_COLORS[s] for s in seg_counts.index], edgecolor="black")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                    f"{int(bar.get_height()):,}", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylabel("Count")
        ax.set_title("Customers per Segment", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Default Rate per Segment")
        seg_def = df2.groupby("RiskSegment")["Default"].mean() * 100
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(seg_def.index, seg_def.values,
                      color=[RISK_COLORS[s] for s in seg_def.index], edgecolor="black")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}%", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylabel("Default Rate (%)")
        ax.set_title("Default Rate by Segment", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    sample = df2.sample(3000, random_state=42)

    st.subheader("Credit Score vs DTI Ratio")
    fig, ax = plt.subplots(figsize=(10, 5))
    for seg, color in RISK_COLORS.items():
        s = sample[sample["RiskSegment"] == seg]
        ax.scatter(s["CreditScore"], s["DTIRatio"], label=seg, alpha=0.25, s=6, color=color)
    ax.set_xlabel("Credit Score"); ax.set_ylabel("DTI Ratio")
    ax.set_title("Customer Segments (KMeans k=3)", fontweight="bold")
    ax.legend(markerscale=6)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Income vs Interest Rate")
    fig, ax = plt.subplots(figsize=(10, 5))
    for seg, color in RISK_COLORS.items():
        s = sample[sample["RiskSegment"] == seg]
        ax.scatter(s["Income"], s["InterestRate"], label=seg, alpha=0.25, s=6, color=color)
    ax.set_xlabel("Income"); ax.set_ylabel("Interest Rate (%)")
    ax.set_title("Income vs Interest Rate by Segment", fontweight="bold")
    ax.legend(markerscale=6)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════
# PAGE 4 — TOPSIS
# ═══════════════════════════════════════════════════════
elif page == "📐 TOPSIS Scoring":
    st.title("📐 TOPSIS Multi-Criteria Risk Scoring")
    st.markdown("""
    **TOPSIS** ranks every customer using 4 equal-weight criteria (0.25 each):

    | Criterion | Direction |
    |-----------|-----------|
    | Credit Score | ✅ Higher is better |
    | DTI Ratio | ❌ Lower is better |
    | Interest Rate | ❌ Lower is better |
    | Income | ✅ Higher is better |

    **Score → 1 = Safest &nbsp;&nbsp;|&nbsp;&nbsp; Score → 0 = Riskiest**
    """)
    st.markdown("---")

    st.subheader("TOPSIS Score Distribution by Risk Segment")
    fig, ax = plt.subplots(figsize=(9, 4))
    for seg, color in RISK_COLORS.items():
        sns.kdeplot(df2[df2["RiskSegment"] == seg]["TOPSIS_Score"],
                    label=seg, color=color, fill=True, alpha=0.3, ax=ax)
    ax.set_xlabel("TOPSIS Score (higher = safer)")
    ax.set_title("Score Distribution", fontweight="bold")
    ax.legend()
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    show_cols = ["CreditScore", "DTIRatio", "Income", "InterestRate",
                 "Default", "RiskSegment", "TOPSIS_Score"]
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔴 Top 10 Riskiest Customers")
        st.dataframe(
            df2.sort_values("TOPSIS_Score")[show_cols].head(10).round(4),
            use_container_width=True)

    with col2:
        st.subheader("🟢 Top 10 Safest Customers")
        st.dataframe(
            df2.sort_values("TOPSIS_Score", ascending=False)[show_cols].head(10).round(4),
            use_container_width=True)

    st.markdown("---")
    st.subheader("TOPSIS Score vs Credit Score")
    sample = df2.sample(3000, random_state=42)
    fig, ax = plt.subplots(figsize=(9, 5))
    for seg, color in RISK_COLORS.items():
        s = sample[sample["RiskSegment"] == seg]
        ax.scatter(s["CreditScore"], s["TOPSIS_Score"],
                   label=seg, alpha=0.35, s=8, color=color)
    ax.set_xlabel("Credit Score"); ax.set_ylabel("TOPSIS Score")
    ax.set_title("Credit Score vs TOPSIS Risk Score", fontweight="bold")
    ax.legend(markerscale=4)
    plt.tight_layout(); st.pyplot(fig); plt.close()

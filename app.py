import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BNPL Default Risk Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

sns.set_theme(style="whitegrid", palette="muted")

# ─────────────────────────────────────────────
# Load Artifacts
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open("label_map.pkl", "rb") as f:
        label_map = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, scaler, kmeans, label_map, feature_names, encoders

@st.cache_data
def load_data():
    df = pd.read_csv("bnpl_sample_40000.csv")
    return df

model, scaler, kmeans, label_map, feature_names, encoders = load_artifacts()
df = load_data()

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/bank-building.png", width=80)
st.sidebar.title("🏦 BNPL Risk Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview & EDA", "🔍 Predict Risk", "📌 Customer Segments", "📐 TOPSIS Scoring"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Dataset: 40,000 BNPL Loan Records")
st.sidebar.caption("Model: Random Forest + SMOTE")

# ─────────────────────────────────────────────
# Helper: Preprocess
# ─────────────────────────────────────────────
cat_cols = ["Education", "EmploymentType", "MaritalStatus",
            "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner"]

@st.cache_data
def get_processed_df():
    df2 = df.copy()
    df2.drop("LoanID", axis=1, inplace=True)
    for col in cat_cols:
        df2[col] = encoders[col].transform(df2[col])
    X = df2.drop("Default", axis=1)
    X_scaled = scaler.transform(X)
    df2["Cluster"] = kmeans.predict(X_scaled)
    df2["RiskSegment"] = df2["Cluster"].map(label_map)
    cluster_summary = df2.groupby("Cluster")[
        ["CreditScore", "DTIRatio", "Income", "InterestRate", "Default"]
    ].mean().round(3)
    # TOPSIS
    topsis_feats = ["CreditScore", "DTIRatio", "InterestRate", "Income"]
    matrix = df2[topsis_feats].values.astype(float)
    norm = matrix / np.sqrt((matrix**2).sum(axis=0))
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    weighted = norm * weights
    benefit = [True, False, False, True]
    ideal_best  = np.where(benefit, weighted.max(0), weighted.min(0))
    ideal_worst = np.where(benefit, weighted.min(0), weighted.max(0))
    d_best  = np.sqrt(((weighted - ideal_best)**2).sum(axis=1))
    d_worst = np.sqrt(((weighted - ideal_worst)**2).sum(axis=1))
    df2["TOPSIS_Score"] = d_worst / (d_best + d_worst)
    return df2, cluster_summary

df2, cluster_summary = get_processed_df()

RISK_COLORS = {"Low Risk": "#2ecc71", "Medium Risk": "#f39c12", "High Risk": "#e74c3c"}

# ═══════════════════════════════════════════
# PAGE 1 — Overview & EDA
# ═══════════════════════════════════════════
if page == "📊 Overview & EDA":
    st.title("📊 BNPL Default Risk — Exploratory Data Analysis")
    st.markdown("---")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Default Rate", f"{df['Default'].mean()*100:.2f}%")
    col3.metric("Avg Credit Score", f"{df['CreditScore'].mean():.0f}")
    col4.metric("Avg DTI Ratio", f"{df['DTIRatio'].mean():.3f}")

    st.markdown("---")

    # Default Distribution
    st.subheader("📌 Default Distribution")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df["Default"].value_counts()
        bars = ax.bar(["Safe (0)", "Risky (1)"], counts.values,
                      color=["#2ecc71", "#e74c3c"], edgecolor="black")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                    f"{int(bar.get_height()):,}", ha="center", fontsize=11, fontweight="bold")
        ax.set_title("Count by Class", fontweight="bold")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(counts.values, labels=["Safe", "Risky"], autopct="%1.1f%%",
               colors=["#2ecc71", "#e74c3c"], startangle=90,
               wedgeprops=dict(edgecolor="white"))
        ax.set_title("Proportion", fontweight="bold")
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Feature Distributions
    st.subheader("📉 Numerical Feature Distributions by Default Status")
    num_cols = ["CreditScore", "DTIRatio", "Income", "LoanAmount", "InterestRate", "Age"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.histplot(data=df, x=col, hue="Default", kde=True, ax=axes[i],
                     palette={0: "#2ecc71", 1: "#e74c3c"}, alpha=0.6, bins=40)
        axes[i].set_title(f"{col}", fontweight="bold")
        axes[i].legend(["Safe", "Risky"], title="Default")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Categorical default rates
    st.subheader("📋 Default Rate by Categorical Feature")
    selected_cat = st.selectbox("Select Feature", cat_cols)
    fig, ax = plt.subplots(figsize=(8, 4))
    default_rate = df.groupby(selected_cat)["Default"].mean() * 100
    default_rate.sort_values().plot(kind="bar", ax=ax,
                                    color="#e74c3c", edgecolor="black", alpha=0.8)
    ax.set_title(f"Default Rate by {selected_cat}", fontweight="bold")
    ax.set_ylabel("Default Rate (%)")
    ax.tick_params(axis="x", rotation=30)
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{bar.get_height():.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

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
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Raw data
    with st.expander("🔎 View Raw Data Sample"):
        st.dataframe(df.head(100), use_container_width=True)

# ═══════════════════════════════════════════
# PAGE 2 — Predict Risk
# ═══════════════════════════════════════════
elif page == "🔍 Predict Risk":
    st.title("🔍 Predict Customer Default Risk")
    st.markdown("Enter customer details below to get a real-time risk prediction.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Financial")
        credit_score    = st.slider("Credit Score", 300, 900, 650)
        income          = st.number_input("Annual Income (₹)", min_value=1000, max_value=500000, value=50000, step=1000)
        loan_amount     = st.number_input("Loan Amount (₹)", min_value=1000, max_value=200000, value=20000, step=1000)
        interest_rate   = st.slider("Interest Rate (%)", 0.0, 30.0, 10.0, 0.1)
        dti_ratio       = st.slider("DTI Ratio", 0.0, 1.0, 0.3, 0.01)

    with col2:
        st.subheader("Personal")
        age             = st.slider("Age", 18, 75, 35)
        education       = st.selectbox("Education", encoders["Education"].classes_)
        marital_status  = st.selectbox("Marital Status", encoders["MaritalStatus"].classes_)
        has_mortgage    = st.selectbox("Has Mortgage", encoders["HasMortgage"].classes_)
        has_dependents  = st.selectbox("Has Dependents", encoders["HasDependents"].classes_)

    with col3:
        st.subheader("Employment & Loan")
        employment_type = st.selectbox("Employment Type", encoders["EmploymentType"].classes_)
        months_employed = st.slider("Months Employed", 0, 120, 36)
        num_credit_lines = st.slider("Num Credit Lines", 1, 10, 3)
        loan_term       = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
        loan_purpose    = st.selectbox("Loan Purpose", encoders["LoanPurpose"].classes_)
        has_cosigner    = st.selectbox("Has Co-Signer", encoders["HasCoSigner"].classes_)

    st.markdown("---")

    if st.button("🚀 Predict Risk", use_container_width=True):
        # Build input row
        input_dict = {
            "Age": age,
            "Income": income,
            "LoanAmount": loan_amount,
            "CreditScore": credit_score,
            "MonthsEmployed": months_employed,
            "NumCreditLines": num_credit_lines,
            "InterestRate": interest_rate,
            "LoanTerm": loan_term,
            "DTIRatio": dti_ratio,
            "Education": encoders["Education"].transform([education])[0],
            "EmploymentType": encoders["EmploymentType"].transform([employment_type])[0],
            "MaritalStatus": encoders["MaritalStatus"].transform([marital_status])[0],
            "HasMortgage": encoders["HasMortgage"].transform([has_mortgage])[0],
            "HasDependents": encoders["HasDependents"].transform([has_dependents])[0],
            "LoanPurpose": encoders["LoanPurpose"].transform([loan_purpose])[0],
            "HasCoSigner": encoders["HasCoSigner"].transform([has_cosigner])[0],
        }

        input_df = pd.DataFrame([input_dict])[feature_names]
        input_scaled = scaler.transform(input_df)

        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        cluster = kmeans.predict(input_scaled)[0]
        risk_seg = label_map[cluster]

        # Results
        c1, c2, c3 = st.columns(3)
        with c1:
            if pred == 1:
                st.error(f"⚠️ **RISKY — Likely to Default**")
            else:
                st.success(f"✅ **SAFE — Unlikely to Default**")
        with c2:
            st.metric("Default Probability", f"{prob*100:.1f}%")
        with c3:
            color = RISK_COLORS.get(risk_seg, "gray")
            st.markdown(f"**Risk Segment:** <span style='color:{color};font-weight:bold'>{risk_seg}</span>",
                        unsafe_allow_html=True)

        # Probability gauge
        fig, ax = plt.subplots(figsize=(6, 1.5))
        bar_color = "#e74c3c" if prob > 0.5 else "#2ecc71"
        ax.barh(["Default Probability"], [prob], color=bar_color, height=0.5)
        ax.barh(["Default Probability"], [1 - prob], left=[prob], color="#ecf0f1", height=0.5)
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.text(prob / 2, 0, f"{prob*100:.1f}%", ha="center", va="center",
                color="white", fontweight="bold", fontsize=12)
        ax.set_title("Default Probability", fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ═══════════════════════════════════════════
# PAGE 3 — Customer Segments
# ═══════════════════════════════════════════
elif page == "📌 Customer Segments":
    st.title("📌 KMeans Customer Segmentation")
    st.markdown("Customers are grouped into 3 risk clusters using KMeans clustering.")
    st.markdown("---")

    # Cluster summary
    st.subheader("Cluster Summary")
    st.dataframe(
        cluster_summary.style.background_gradient(cmap="RdYlGn_r", subset=["Default"]),
        use_container_width=True
    )

    st.markdown("---")

    # Segment distribution
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Risk Segment Distribution")
        seg_counts = df2["RiskSegment"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(seg_counts.index, seg_counts.values,
                      color=[RISK_COLORS[s] for s in seg_counts.index], edgecolor="black")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f"{int(bar.get_height()):,}", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylabel("Count")
        ax.set_title("Segment Counts", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Default Rate per Segment")
        seg_default = df2.groupby("RiskSegment")["Default"].mean() * 100
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(seg_default.index, seg_default.values,
                      color=[RISK_COLORS[s] for s in seg_default.index], edgecolor="black")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}%", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylabel("Default Rate (%)")
        ax.set_title("Default Rate by Segment", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Scatter
    st.subheader("Credit Score vs DTI Ratio (by Risk Segment)")
    fig, ax = plt.subplots(figsize=(10, 5))
    for seg, color in RISK_COLORS.items():
        subset = df2[df2["RiskSegment"] == seg]
        ax.scatter(subset["CreditScore"], subset["DTIRatio"],
                   label=seg, alpha=0.2, s=5, color=color)
    ax.set_xlabel("Credit Score")
    ax.set_ylabel("DTI Ratio")
    ax.set_title("Customer Segments", fontweight="bold")
    ax.legend(markerscale=6)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ═══════════════════════════════════════════
# PAGE 4 — TOPSIS
# ═══════════════════════════════════════════
elif page == "📐 TOPSIS Scoring":
    st.title("📐 TOPSIS Multi-Criteria Risk Scoring")
    st.markdown("""
    **TOPSIS** (Technique for Order of Preference by Similarity to Ideal Solution)  
    ranks each customer based on 4 criteria:
    - **CreditScore** — higher is better ✅
    - **DTI Ratio** — lower is better ❌
    - **Interest Rate** — lower is better ❌
    - **Income** — higher is better ✅
    
    Score close to **1 = Safest**, close to **0 = Riskiest**
    """)
    st.markdown("---")

    # Distribution
    st.subheader("TOPSIS Score Distribution by Risk Segment")
    fig, ax = plt.subplots(figsize=(9, 4))
    for seg, color in RISK_COLORS.items():
        subset = df2[df2["RiskSegment"] == seg]["TOPSIS_Score"]
        sns.kdeplot(subset, label=seg, color=color, fill=True, alpha=0.3, ax=ax)
    ax.set_xlabel("TOPSIS Score (higher = safer)")
    ax.set_title("Score Distribution", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔴 Top 10 Riskiest Customers")
        risky = df2.sort_values("TOPSIS_Score")[
            ["CreditScore", "DTIRatio", "Income", "InterestRate", "Default", "RiskSegment", "TOPSIS_Score"]
        ].head(10).round(4)
        st.dataframe(risky, use_container_width=True)

    with col2:
        st.subheader("🟢 Top 10 Safest Customers")
        safest = df2.sort_values("TOPSIS_Score", ascending=False)[
            ["CreditScore", "DTIRatio", "Income", "InterestRate", "Default", "RiskSegment", "TOPSIS_Score"]
        ].head(10).round(4)
        st.dataframe(safest, use_container_width=True)

    st.markdown("---")

    # Scatter: TOPSIS vs CreditScore
    st.subheader("TOPSIS Score vs Credit Score")
    sample = df2.sample(3000, random_state=42)
    fig, ax = plt.subplots(figsize=(9, 5))
    for seg, color in RISK_COLORS.items():
        subset = sample[sample["RiskSegment"] == seg]
        ax.scatter(subset["CreditScore"], subset["TOPSIS_Score"],
                   label=seg, alpha=0.4, s=8, color=color)
    ax.set_xlabel("Credit Score")
    ax.set_ylabel("TOPSIS Score")
    ax.set_title("Credit Score vs TOPSIS Risk Score", fontweight="bold")
    ax.legend(markerscale=4)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

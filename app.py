import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter

# ==============================
# SIDEBAR
# ==============================

st.sidebar.title("🧬 METABRIC Analysis")

st.sidebar.markdown(
    "### 📋 Navigation"
)

page = st.sidebar.radio(
    "Select a section:",
    [
        "🏠 Overview",
        "📊 EDA",
        "🔮 Mortality Prediction",
        "📈 Model Performance",
        "🧬 Survival Analysis"
    ],
    index=0
)

st.sidebar.markdown("---")

st.sidebar.markdown("### 📌 Project Information")

st.sidebar.write(
    "**Dataset:** METABRIC"
)

st.sidebar.write(
    "**Task:** 10-Year Mortality Prediction"
)

st.sidebar.write(
    "**Models:** Logistic Regression, Decision Tree, SVM"
)

st.sidebar.write(
    "**Survival:** Kaplan-Meier & Cox Regression"
)

st.sidebar.markdown("---")

st.sidebar.success(
    f"Current Section: {page}"
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "🧬 Breast Cancer Survival & Mortality Analysis"
)

st.sidebar.caption(
    "Use the menu above to explore the project."
)

# Load dataset

df = pd.read_csv("cleaned_metabric_data.csv")

# Load trained models
logistic_model = joblib.load("logistic_model.pkl")
svm_model = joblib.load("svm_model.pkl")
decision_tree = joblib.load("decision_tree.pkl")

# Load preprocessing objects
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
feature_names = joblib.load("feature_names.pkl")

# Title
if page == "🏠 Overview":
    st.title("🧬 METABRIC Breast Cancer Analysis")

st.write(
    """
    Machine Learning Application for 10-Year Mortality Risk Prediction
    using the METABRIC breast cancer dataset.
    """
)

# Dataset overview
st.subheader("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Patients", df.shape[0])

with col2:
    st.metric("Features", df.shape[1] - 1)

with col3:
    st.metric("Target", "10-Year Mortality")

# Dataset preview
st.subheader("📋 Dataset Preview")

st.dataframe(
    df.head(10),
    width="stretch"
)
# Models
st.subheader("🤖 Machine Learning Models")

st.write(
    """
    The application uses three classification algorithms:

    • Logistic Regression

    • Support Vector Machine (SVM)

    • Decision Tree
    """
)
st.success("Models and preprocessing objects loaded successfully!")

# ---------------------------------------------------------
# 10-Year Mortality Prediction
# ---------------------------------------------------------
if page == "🔮 Mortality Prediction":
    st.subheader("🔮 10-Year Mortality Risk Prediction")

st.write(
    "Enter patient information below to estimate the 10-year mortality risk."
)

# Numerical inputs
age = st.number_input(
    "Age at Diagnosis",
    min_value=18.0,
    max_value=100.0,
    value=55.0
)

lymph_nodes = st.number_input(
    "Lymph nodes examined positive",
    min_value=0.0,
    value=0.0
)

mutation_count = st.number_input(
    "Mutation Count",
    min_value=0.0,
    value=5.0
)

npi = st.number_input(
    "Nottingham prognostic index",
    min_value=0.0,
    value=3.0
)

tumor_size = st.number_input(
    "Tumor Size",
    min_value=0.0,
    value=20.0
)

tumor_stage = st.number_input(
    "Tumor Stage",
    min_value=0.0,
    value=2.0
)

st.write("Click the button below to make a prediction.")
if st.button("🔍 Predict Mortality Risk"):

    patient_data = pd.DataFrame({
        "Age at Diagnosis": [age],
        "Lymph nodes examined positive": [lymph_nodes],
        "Mutation Count": [mutation_count],
        "Nottingham prognostic index": [npi],
        "Tumor Size": [tumor_size],
        "Tumor Stage": [tumor_stage]
    })

    # Add remaining model features
    for feature in feature_names:
        if feature not in patient_data.columns:
            patient_data[feature] = 0

    # Keep the same feature order used during training
    patient_data = patient_data[feature_names]

    # Scale for Logistic Regression and SVM
    patient_scaled = scaler.transform(patient_data)

    # Predictions
    lr_prediction = logistic_model.predict(patient_scaled)[0]
    svm_prediction = svm_model.predict(patient_scaled)[0]
    dt_prediction = decision_tree.predict(patient_data)[0]

    # Probabilities
    lr_probability = logistic_model.predict_proba(patient_scaled)[0][1]
    svm_probability = svm_model.predict_proba(patient_scaled)[0][1]
    dt_probability = decision_tree.predict_proba(patient_data)[0][1]

    st.subheader("📊 Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("### Logistic Regression")

        if lr_prediction == 1:
            st.error("Higher Risk")
        else:
            st.success("Lower Risk")

        st.metric(
            "Mortality Probability",
            f"{lr_probability * 100:.2f}%"
        )

    with col2:
        st.write("### SVM")

        if svm_prediction == 1:
            st.error("Higher Risk")
        else:
            st.success("Lower Risk")

        st.metric(
            "Mortality Probability",
            f"{svm_probability * 100:.2f}%"
        )

    with col3:
        st.write("### Decision Tree")

        if dt_prediction == 1:
            st.error("Higher Risk")
        else:
            st.success("Lower Risk")

        st.metric(
            "Mortality Probability",
            f"{dt_probability * 100:.2f}%"
        )

    st.info(
        "These predictions are generated by machine-learning models "
        "trained on the project dataset and are intended for demonstration purposes."
    )
# ---------------------------------------------------------
# Model Performance
# ---------------------------------------------------------

model_results = joblib.load("model_results.pkl")

st.subheader("📈 Model Performance")

st.dataframe(
    model_results,
    width="stretch"
)

st.subheader("📊 Accuracy Comparison")

st.bar_chart(
    model_results.set_index("Model")["Accuracy"]
)

st.subheader("📊 ROC-AUC Comparison")

st.bar_chart(
    model_results.set_index("Model")["ROC-AUC"]
)
# ---------------------------------------------------------
# Exploratory Data Analysis
# ---------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns

st.subheader("📊 Exploratory Data Analysis")

# Target distribution
fig1, ax1 = plt.subplots()

sns.countplot(
    data=df,
    x="Target_10yr_Mortality",
    ax=ax1
)

ax1.set_title("10-Year Mortality Distribution")
ax1.set_xlabel("10-Year Mortality")
ax1.set_ylabel("Number of Patients")

st.pyplot(fig1)

# Age distribution
fig2, ax2 = plt.subplots()

sns.histplot(
    data=df,
    x="Age at Diagnosis",
    kde=True,
    ax=ax2
)

ax2.set_title("Age at Diagnosis Distribution")
ax2.set_xlabel("Age at Diagnosis")
ax2.set_ylabel("Number of Patients")

st.pyplot(fig2)

# Tumor size distribution
fig3, ax3 = plt.subplots()

sns.histplot(
    data=df,
    x="Tumor Size",
    kde=True,
    ax=ax3
)

ax3.set_title("Tumor Size Distribution")
ax3.set_xlabel("Tumor Size")
ax3.set_ylabel("Number of Patients")

st.pyplot(fig3)

# Mortality by tumor stage
fig4, ax4 = plt.subplots()

sns.countplot(
    data=df,
    x="Tumor Stage",
    hue="Target_10yr_Mortality",
    ax=ax4
)

ax4.set_title("10-Year Mortality by Tumor Stage")
ax4.set_xlabel("Tumor Stage")
ax4.set_ylabel("Number of Patients")

st.pyplot(fig4)

# ==============================
# SURVIVAL ANALYSIS
# ==============================
if page == "🧬 Survival Analysis":
    st.header("🧬 Survival Analysis")

survival_time_col = "Synthetic_Survival_Months"
survival_event_col = "Synthetic_Event"

if survival_time_col in df.columns and survival_event_col in df.columns:

    survival_data = df[
        [survival_time_col, survival_event_col]
    ].dropna()

    survival_data[survival_time_col] = pd.to_numeric(
        survival_data[survival_time_col],
        errors="coerce"
    )

    survival_data[survival_event_col] = pd.to_numeric(
        survival_data[survival_event_col],
        errors="coerce"
    )

    survival_data = survival_data.dropna()

    st.write(
        f"**Patients included in survival analysis:** "
        f"{len(survival_data)}"
    )

    st.write(
        f"**Events:** "
        f"{int(survival_data[survival_event_col].sum())}"
    )

    st.write(
        f"**Median survival time:** "
        f"{survival_data[survival_time_col].median():.2f} months"
    )

else:
    st.error(
        "The dataset does not contain the required survival columns."
    )

 
# -------------------------------------------------
# Cox Proportional Hazards Analysis
# -------------------------------------------------

st.subheader("📌 Cox Proportional Hazards Analysis")

cox_features = [
    "Age at Diagnosis",
    "Lymph nodes examined positive",
    "Mutation Count",
    "Nottingham prognostic index",
    "Tumor Size",
    "Tumor Stage"
]

available_features = [
    col for col in cox_features
    if col in df.columns
]

cox_df = df[
    available_features +
    [survival_time_col, survival_event_col]
].copy()

# Convert numeric variables
for col in available_features:
    cox_df[col] = pd.to_numeric(
        cox_df[col],
        errors="coerce"
    )

# Convert survival time
cox_df[survival_time_col] = pd.to_numeric(
    cox_df[survival_time_col],
    errors="coerce"
)

# Convert survival event
cox_df["Event"] = pd.to_numeric(
    cox_df[survival_event_col],
    errors="coerce"
)

# Remove original event column
cox_df = cox_df.drop(
    columns=[survival_event_col]
)

# Remove missing values
cox_df = cox_df.dropna()

# Remove invalid survival times
cox_df = cox_df[
    cox_df[survival_time_col] > 0
]

if len(cox_df) > 0:

    cph = CoxPHFitter(penalizer=0.1)

    cph.fit(
        cox_df,
        duration_col=survival_time_col,
        event_col="Event"
    )

    st.write("### Cox Regression Results")

    st.dataframe(
        cph.summary
    )

else:

    st.warning(
        "Not enough valid survival data available "
        "for Cox Proportional Hazards analysis."
    )
# -------------------------------------------------
# Kaplan-Meier Survival Analysis
# -------------------------------------------------

st.subheader("📈 Kaplan-Meier Survival Curve")

kmf = KaplanMeierFitter()

kmf.fit(
    durations=survival_data[survival_time_col],
    event_observed=survival_data[survival_event_col]
)

fig_km, ax_km = plt.subplots(figsize=(8, 5))

kmf.plot_survival_function(ax=ax_km)

ax_km.set_title("Kaplan-Meier Survival Curve")
ax_km.set_xlabel("Survival Time (Months)")
ax_km.set_ylabel("Survival Probability")

ax_km.grid(alpha=0.3)

st.pyplot(fig_km)

st.write(
    f"**Median survival time:** "
    f"{kmf.median_survival_time_:.2f} months"
)
# -------------------------------------------------
# Kaplan-Meier Curve by 10-Year Mortality
# -------------------------------------------------

st.subheader("📊 Survival by 10-Year Mortality Group")

km_group_data = df[
    [
        survival_time_col,
        survival_event_col,
        "Target_10yr_Mortality"
    ]
].copy()

km_group_data[survival_time_col] = pd.to_numeric(
    km_group_data[survival_time_col],
    errors="coerce"
)

km_group_data[survival_event_col] = pd.to_numeric(
    km_group_data[survival_event_col],
    errors="coerce"
)

km_group_data["Target_10yr_Mortality"] = pd.to_numeric(
    km_group_data["Target_10yr_Mortality"],
    errors="coerce"
)

km_group_data = km_group_data.dropna()

km_group_data = km_group_data[
    km_group_data[survival_time_col] > 0
]

fig_group, ax_group = plt.subplots(figsize=(8, 5))

kmf = KaplanMeierFitter()

for group in sorted(
    km_group_data["Target_10yr_Mortality"].unique()
):

    group_data = km_group_data[
        km_group_data["Target_10yr_Mortality"] == group
    ]

    kmf.fit(
        durations=group_data[survival_time_col],
        event_observed=group_data[survival_event_col],
        label=f"Mortality Group {int(group)}"
    )

    kmf.plot_survival_function(ax=ax_group)

ax_group.set_title(
    "Kaplan-Meier Survival by 10-Year Mortality Group"
)

ax_group.set_xlabel("Survival Time (Months)")
ax_group.set_ylabel("Survival Probability")

ax_group.grid(alpha=0.3)

st.pyplot(fig_group)



🧬 METABRIC Breast Cancer Analysis

📌 Project Overview

This project is a machine learning and survival analysis application based on the METABRIC breast cancer dataset.

The application provides exploratory data analysis, data preprocessing, machine learning-based 10-year mortality risk prediction, model evaluation, and survival analysis through an interactive Streamlit web application.

🌐 Live Streamlit Application

Open the deployed application here:

https://breast-cancer-predictor-c87svtsrycp5w3j8kreqgt.streamlit.app/

🎯 Project Objectives

* Analyze breast cancer patient data
* Perform exploratory data analysis (EDA)
* Preprocess clinical and categorical data
* Predict 10-year mortality risk
* Compare multiple machine learning models
* Evaluate model performance
* Perform Kaplan-Meier survival analysis
* Perform Cox Proportional Hazards analysis

🤖 Machine Learning Models

The application uses three classification algorithms:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Decision Tree

📊 Model Evaluation

The models are evaluated using performance measures including:

* Accuracy
* ROC-AUC
* Classification performance

The application also provides visual comparisons of model performance.

🧬 Survival Analysis

The application includes:

* Kaplan-Meier overall survival curve
* Kaplan-Meier survival curves by 10-year mortality group
* Cox Proportional Hazards regression

Note: The available cleaned dataset uses Synthetic_Survival_Months and Synthetic_Event for the survival-analysis component.

📁 Project Files

METABRIC_PROJECT/
│
├── app.py
├── cleaned_metabric_data.csv
├── requirements.txt
│
├── categorical_cols.pkl
├── categorical_imputer.pkl
├── decision_tree.pkl
├── encoder.pkl
├── feature_names.pkl
├── logistic_model.pkl
├── model_results.pkl
├── numerical_cols.pkl
├── numerical_imputer.pkl
├── scaler.pkl
└── svm_model.pkl

💻 How to Run the Application Locally

1. Install the required packages

pip install -r requirements.txt

2. Run the Streamlit application

python3 -m streamlit run app.py

3. Open the application

Streamlit will provide a local URL such as:

http://localhost:8501

Open this URL in your web browser.

🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Lifelines
* Streamlit

👩‍💻 Project Author

Afifa Alam

📌 Note

This application is developed for academic/project purposes using the METABRIC breast cancer dataset.

# 🏦 Loan Approval Predictor

This project is a complete machine learning pipeline that predicts loan approval status based on applicant information. It includes:

- ✅ Exploratory Data Analysis (EDA)
- 🧹 Data Preprocessing Module
- 📊 Multiple Model Training & Comparison
- 🏆 Logistic Regression selected as the best model
- 💡 Streamlit App for user input-based prediction
- 📈 Interactive Explainer Dashboard to understand model decisions

---

## 📂 Project Structure

```
loan_approval_predictor/
│
├── app/
│   └── streamlit_app.py         # Streamlit web app
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── model_training.ipynb     # Model comparison
├── src/
│   └── data_preprocessing.py    # Reusable cleaning module
├── model/
│   ├── model.joblib             # Trained logistic regression model
│   └── test_data.joblib         # X_test, y_test
├── requirements.txt
└── README.md
```

---

## 🚀 Live Dashboard

🔍 Explore the model explanations interactively using SHAP, What-If analysis, and more:

👉 [**Loan Approval Explainer Dashboard**](https://loan-approval-predictor-explainer.onrender.com)

---

## 🖥️ Run the Streamlit App Locally

### 📌 Prerequisites

- Python 3.8+
- `pip install -r requirements.txt`

### ▶️ Run App

```bash
cd app
streamlit run streamlit_app.py
```

It will launch a web UI to input user details and check loan approval prediction.

---

## 📂 Dataset

The dataset used in this project contains information related to loan applicants such as gender, income, credit history, loan amount, and property details.

- **Source**: [Loan Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- **Total Records**: 614 entries
- **Features**:
  - Gender, Marital Status, Dependents
  - Education, Self_Employed, ApplicantIncome
  - LoanAmount, Loan_Amount_Term, Credit_History
  - Property_Area, Loan_Status (Target)

The dataset was cleaned and preprocessed using the custom `load_and_clean_data()` function in the `src/data_preprocessing.py` module.


---

## 📊 Model Comparison Summary

We trained and evaluated multiple models:
- Logistic Regression ✅ *(Best Accuracy)*
- Decision Tree
- Random Forest
- KNN
- XGBoost

Comparison is visualized in `notebooks/model_training.ipynb`.

---

## 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📌 Tech Stack

- Python, Pandas, scikit-learn, XGBoost
- Matplotlib, Seaborn (for EDA)
- Streamlit (user interface)
- ExplainerDashboard (model interpretability)
- Render (deployment)

---

## 🤝 Contribution & Credits

This project was to demonstrate the end-to-end lifecycle of a ML application. Contributions and forks are welcome!

---

## 🧠 Author

👤 Somesh Chaurasia  
🎓 B.Tech CSE

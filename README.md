# 💸 Loan Approval Predictor

A machine learning-based tool to predict loan approval status using user details. It includes:

- 🧪 Exploratory Data Analysis (EDA)
- 🧹 Data Cleaning using a reusable Python module
- 🤖 Multiple ML models with performance comparison
- 📊 SHAP-based Explainability Dashboard
- 🧑‍💼 Streamlit App for live user predictions

---

## 🚀 Project Structure

```
loan_approval_predictor/
│
├── app/
│   └── streamlit_app.py          # Streamlit app for user prediction
│
├── data/
│   └── loan.csv                  # Raw dataset
│
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── model_training.ipynb
│
├── src/
│   └── data_preprocessing.py     # Reusable data cleaning functions
│
├── model.joblib                  # Saved best-performing model
├── test_data.joblib              # X_test, y_test for ExplainerDashboard
├── explainer_dashboard.py        # ExplainerDashboard code
├── requirements.txt
└── README.md
```

---

## 📈 ML Model Comparison

We trained and evaluated multiple models:
- Logistic Regression ✅ (Best)
- Decision Tree
- Random Forest
- XGBoost

Final model chosen: **Logistic Regression** based on highest test accuracy.

---

## 📊 Explainability

We used `ExplainerDashboard` with SHAP to interpret model predictions.

### 🔗 Live Dashboard:  
👉 [Click here to view](https://your-deployed-dashboard-link.com) 

---

## 🧠 Top Influential Features (from SHAP)

1. Credit_History  
2. Married  
3. Education  
4. Gender  
5. LoanAmount  

---

## 🖥️ Streamlit App (for Demo)

Use this locally to test loan approval by entering key user details.

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/streamlit_app.py
```

---

## 🌐 Deployment Info

- ExplainerDashboard deployed on: **Heroku/Render** *(add once deployed)*
- Streamlit app is local-only for now; clone this repo to run

---

## ✍️ How to Contribute

Clone the repo and follow the structure. You can modify or add new models and retrain.

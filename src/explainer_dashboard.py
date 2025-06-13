import joblib
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

# Loading model and test data
model = joblib.load("model.joblib")
X_test, y_test = joblib.load("test_data.joblib")

# Create explainer obj
explainer = ClassifierExplainer(model, X_test, y_test)

# Create dashboard gui
db = ExplainerDashboard(explainer,
                        title="Loan Approval Explainer Dashboard",
                        whatif=True,
                        shap_interaction=True,
                        )

# Run dashboard, entry
if __name__ == "__main__":
    db.run(port=8050)

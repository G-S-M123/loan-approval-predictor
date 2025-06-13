import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from data_preprocessing import load_and_clean_data

def train_and_save_model():
    X_train, X_test, y_train, y_test = load_and_clean_data()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save model and test data
    joblib.dump(model, "model.joblib")
    joblib.dump((X_test, y_test), "test_data.joblib")

if __name__ == "__main__":
    train_and_save_model()

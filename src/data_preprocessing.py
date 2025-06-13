import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(path='../data/loan.csv'):
    df = pd.read_csv(path)

    # Drop Loan_ID (not useful for prediction)
    df.drop('Loan_ID', axis=1, inplace=True)

    # Fill missing values
    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']:
        df[col].fillna(df[col].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)

    # Encode categorical columns
    label_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in label_cols:
        df[col] = le.fit_transform(df[col])

    # Split features and target
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

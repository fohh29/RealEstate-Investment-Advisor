#. data preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Configuration
RAW_DATA_PATH = "data/raw/india_housing_prices.csv"
PROCESSED_DATA_PATH = "data/processed/"
MODEL_PATH = "models/"

os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

def preprocess():
    df = pd.read_csv(RAW_DATA_PATH)
    
    # 1. Target Generation
    np.random.seed(42)
    cities = df['City'].unique()
    growth_rates = {city: np.random.uniform(0.05, 0.12) for city in cities}
    df['Growth_Rate'] = df['City'].map(growth_rates)
    df['Future_Price_5Y'] = df['Price_in_Lakhs'] * (1 + df['Growth_Rate'])**5
    df['ROI_5Y'] = (df['Future_Price_5Y'] - df['Price_in_Lakhs']) / df['Price_in_Lakhs']
    df['Good_Investment'] = (df['ROI_5Y'] > 0.40).astype(int)

    # 2. Encoding categorical columns
    # We find all columns that are 'object' (text) and convert them to numbers
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        # Save the encoder for the app to use later
        joblib.dump(le, f"{MODEL_PATH}{col.lower()}_encoder.pkl")
        print(f"✅ Saved encoder for: {col}")

    # 3. Save clean data
    cols_to_drop = ['ID', 'Growth_Rate', 'ROI_5Y']
    df_clean = df.drop(columns=cols_to_drop)
    
    X = df_clean.drop(columns=['Future_Price_5Y', 'Good_Investment'])
    y_reg = df_clean['Future_Price_5Y']
    y_clf = df_clean['Good_Investment']
    
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )
    
    # Saving all 6 files
    X_train.to_csv(f"{PROCESSED_DATA_PATH}X_train.csv", index=False)
    X_test.to_csv(f"{PROCESSED_DATA_PATH}X_test.csv", index=False)
    y_reg_train.to_csv(f"{PROCESSED_DATA_PATH}y_reg_train.csv", index=False)
    y_reg_test.to_csv(f"{PROCESSED_DATA_PATH}y_reg_test.csv", index=False)
    y_clf_train.to_csv(f"{PROCESSED_DATA_PATH}y_clf_train.csv", index=False)
    y_clf_test.to_csv(f"{PROCESSED_DATA_PATH}y_clf_test.csv", index=False)
    
    print("\n✅ Preprocessing Complete! Check your folders now.")

if __name__ == "__main__":
    preprocess()
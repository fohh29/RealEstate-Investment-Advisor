# Import library
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- Import Classifiers ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# --- Import Regressors ---
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


# Load Processed Data

try:
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_reg_train = pd.read_csv("data/processed/y_reg_train.csv").values.ravel()
    y_clf_train = pd.read_csv("data/processed/y_clf_train.csv").values.ravel()
    X_train = pd.DataFrame(X_train).sample(n=10000, random_state=42)
    y_clf_train = y_clf_train[X_train.index]
    y_reg_train = y_reg_train[X_train.index]
    
    
    X_test = pd.read_csv("data/processed/X_test.csv")
    
    y_reg_test = pd.read_csv("data/processed/y_reg_test.csv").values.ravel()
    
    y_clf_test = pd.read_csv("data/processed/y_clf_test.csv").values.ravel()
except FileNotFoundError:
    print("❌ Error: Processed data not found. Please run src/data_preprocessing.py first.")
    exit()

# Setup MLflow
mlflow.set_experiment("Real_Estate_Advisor_Comparison")

# ==========================================
# 1. CLASSIFICATION TRAINING (Target: Good_Investment)
# ==========================================
def run_classification_experiments():
    print("\n🔹 Starting Classification Experiments...")
    
    # Dictionary of 5+ Models
    classifiers = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Decision_Tree": DecisionTreeClassifier(max_depth=10),
        "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "Gradient_Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1),
        "SVM": SVC(probability=True) # SVC needs probability=True for ROC_AUC
    }

    best_clf_score = 0
    best_clf_model = None
    best_clf_name = ""

    for name, model in classifiers.items():
        with mlflow.start_run(run_name=f"CLF_{name}"):
            # Create Pipeline 
            pipeline = Pipeline([
                ('scaler', StandardScaler()), 
                ('model', model)
            ])
            
            # Train
            pipeline.fit(X_train, y_clf_train)
            
            # Predict
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline['model'], "predict_proba") else [0]*len(y_pred)
            
            # Metrics
            acc = accuracy_score(y_clf_test, y_pred)
            prec = precision_score(y_clf_test, y_pred, zero_division=0)
            rec = recall_score(y_clf_test, y_pred, zero_division=0)
            f1 = f1_score(y_clf_test, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_clf_test, y_proba)
            except:
                auc = 0
            
            print(f"   -> {name}: Accuracy={acc:.4f}, F1={f1:.4f}")
            
            # Log to MLflow
            mlflow.log_param("model_name", name)
            mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc})
            mlflow.sklearn.log_model(pipeline, f"model_{name}")
            
            # Track Best Model (Optimizing for Accuracy here, can be F1)
            if acc > best_clf_score:
                best_clf_score = acc
                best_clf_model = pipeline
                best_clf_name = name

    # Save Champion Model
    print(f"🏆 Best Classification Model: {best_clf_name} with Accuracy {best_clf_score:.4f}")
    joblib.dump(best_clf_model, "models/investment_classifier.pkl")

# ==========================================
# 2. REGRESSION TRAINING (Target: Future_Price_5Y)
# ==========================================
def run_regression_experiments():
    print("\n🔹 Starting Regression Experiments...")
    
    # Dictionary of 5+ Models
    regressors = {
        "Linear_Regression": LinearRegression(),
        "Ridge_Regression": Ridge(alpha=1.0),
        "Lasso_Regression": Lasso(alpha=0.1),
        "Random_Forest_Reg": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "Gradient_Boosting_Reg": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    best_reg_score = -float("inf") # Using R2, so higher is better
    best_reg_model = None
    best_reg_name = ""

    for name, model in regressors.items():
        with mlflow.start_run(run_name=f"REG_{name}"):
            # Create Pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()), 
                ('model', model)
            ])
            
            # Train
            pipeline.fit(X_train, y_reg_train)
            
            # Predict
            y_pred = pipeline.predict(X_test)
            
            # Metrics
            mae = mean_absolute_error(y_reg_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred))
            r2 = r2_score(y_reg_test, y_pred)
            
            print(f"   -> {name}: RMSE={rmse:.2f}, R2={r2:.4f}")
            
            # Log to MLflow
            mlflow.log_param("model_name", name)
            mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})
            mlflow.sklearn.log_model(pipeline, f"model_{name}")
            
            # Track Best Model (Optimizing for R2)
            if r2 > best_reg_score:
                best_reg_score = r2
                best_reg_model = pipeline
                best_reg_name = name

    # Save Champion Model
    print(f"🏆 Best Regression Model: {best_reg_name} with R2 {best_reg_score:.4f}")
    joblib.dump(best_reg_model, "models/price_regressor.pkl")

if __name__ == "__main__":
    run_classification_experiments()
    run_regression_experiments()
    print("\n✅ All experiments complete. Best models saved to 'models/' folder.")
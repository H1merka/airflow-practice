import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models import infer_signature
import mlflow
import joblib
import os

CLEAR_CSV = "./df_clear.csv"

def scale_frame(frame):
    """
    Принимает уже числовой DataFrame (все фичи + столбец 'Price'),
    возвращает X_scaled, y_scaled, scaler, power_trans.
    """
    df = frame.copy()
    if 'Price' not in df.columns:
        raise ValueError("df must contain target column 'Price'")

    X = df.drop(columns=['Price'])
    y = df['Price'].values.reshape(-1, 1)

    scaler = StandardScaler()
    power_trans = PowerTransformer()  # yeo-johnson по умолчанию (поддерживает отриц.)
    X_scaled = scaler.fit_transform(X.values)
    y_scaled = power_trans.fit_transform(y)

    return X_scaled, y_scaled.reshape(-1), scaler, power_trans

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train():
    if not os.path.exists(CLEAR_CSV):
        raise FileNotFoundError(f"Required cleaned file not found: {CLEAR_CSV}. Run clear_data() first.")

    df = pd.read_csv(CLEAR_CSV)
    print("Training on:", df.shape)

    X, y, scaler, power_trans = scale_frame(df)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    params = {
        'alpha': [0.0001, 0.001, 0.01],
        'l1_ratio': [0.01, 0.1],
        "penalty": ["l2", "elasticnet"],
        "loss": ['squared_error', 'huber'],
        "fit_intercept": [True],
    }

    mlflow.set_experiment("cardekho_linear_sgd")
    with mlflow.start_run():
        base = SGDRegressor(random_state=42, max_iter=10000, tol=1e-4)
        clf = GridSearchCV(base, params, cv=3, n_jobs=4)
        clf.fit(X_train, y_train)
        best = clf.best_estimator_

        # predict on validation (we have y in transformed space)
        y_pred_scaled = best.predict(X_val)
        # inverse transform predictions and actual to original scale for metrics
        y_pred = power_trans.inverse_transform(y_pred_scaled.reshape(-1,1)).reshape(-1)
        y_true = power_trans.inverse_transform(y_val.reshape(-1,1)).reshape(-1)

        rmse, mae, r2 = eval_metrics(y_true, y_pred)

        # Log parameters and metrics
        mlflow.log_param("best_params", clf.best_params_)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # signature (optional) - infer from X_train (unscaled) and predictions
        signature = infer_signature(X_train, best.predict(X_train))

        # Log model to mlflow
        mlflow.sklearn.log_model(best, "model", signature=signature)

        # Save artifacts locally: model, scaler, power transformer
        joblib.dump(best, "best_model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(power_trans, "power_transformer.pkl")

        mlflow.log_artifact("best_model.pkl")
        mlflow.log_artifact("scaler.pkl")
        mlflow.log_artifact("power_transformer.pkl")

    print(f"Training finished. RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return True

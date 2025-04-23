import prepare_data as prepare
import preprocess_data as preprocess
import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, mean_absolute_error, r2_score)
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

############################################ Cancellation Prediction Model ############################################
def train_and_save_models():
    # X and y for cancellation prediction
    X_train_cancel, X_test_cancel, y_train_cancel, y_test_cancel = preprocess.train_test_cancel

    # Train classification model
    preprocess.clf_pipeline.fit(X_train_cancel, y_train_cancel)

    # Predict and evaluate
    y_pred_cancel = preprocess.clf_pipeline.predict(X_test_cancel)
    acc = accuracy_score(y_test_cancel, y_pred_cancel)
    print(f"Cancellation Model Accuracy: {acc * 100:.2f}%")

    # Save cancellation model
    joblib.dump(preprocess.clf_pipeline, "./artifacts/model_cancel.pkl")



    ################################################ ADR Prediction Model #################################################

    # Train a more powerful ADR model
    # Using XGBoost Regressor for better performance
    reg_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )

    # Train on log-transformed ADR
    reg_model.fit(preprocess.X_train_adr, preprocess.y_train_adr)

    # Predict and reverse log1p
    y_pred_log = reg_model.predict(preprocess.X_test_adr)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(preprocess.y_test_adr)

    # Evaluation
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"XGBoost ADR R² Score: {r2:.2f}")
    print(f"XGBoost ADR MAE: {mae:.2f}")

    # Save model
    joblib.dump(reg_model, "./artifacts/model_adr.pkl")


    # Create ADR Features
    # Define ADR features and target
    adr_features = [
        'hotel', 'lead_time', 'adults', 'children', 'babies',
        'stays_in_week_nights', 'stays_in_weekend_nights',
        'is_repeated_guest', 'previous_cancellations',
        'reserved_room_type', 'assigned_room_type', 'deposit_type',
        'meal', 'market_segment', 'distribution_channel',
        'customer_type', 'total_of_special_requests'
    ]

    # Filter out extreme ADR outliers
    adr_df = prepare.df[(prepare.df['adr'] > 0) & (prepare.df['adr'] < 5000)].copy()

    # Select features and target
    X_adr = adr_df[adr_features].copy()
    # Reordering Columns
    adr_feature_order = X_adr.columns.tolist()
    joblib.dump(adr_feature_order, "./artifacts/adr_features.pkl")




    y_adr = np.log1p(adr_df['adr'])  # log-transformed target

    # Label encode categorical columns
    label_encoders = {}
    for col in X_adr.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X_adr[col] = le.fit_transform(X_adr[col])
        label_encoders[col] = le

    # Split the data
    train_test_adr = train_test_split(X_adr, y_adr, test_size=0.2, random_state=42)
    X_train_adr, X_test_adr, y_train_adr, y_test_adr = train_test_adr

    # Save encoders for use in the Dash App (web_app.py)
    joblib.dump(label_encoders, "./artifacts/encoders.pkl")

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
    joblib.dump(preprocess.clf_pipeline, "./artifacts/model_cancel.pkl", compress=3)

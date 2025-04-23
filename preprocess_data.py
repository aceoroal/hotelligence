import prepare_data as prepare
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib

####################################################
# CANCELLATION MODEL SETUP
####################################################

# Define cancellation features and target
cancel_features = [
    'hotel', 'lead_time', 'arrival_date_month', 'arrival_date_week_number',
    'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights',
    'adults', 'children', 'babies', 'meal', 'market_segment', 'distribution_channel',
    'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled',
    'reserved_room_type', 'assigned_room_type', 'booking_changes', 'deposit_type',
    'agent', 'days_in_waiting_list', 'customer_type', 'required_car_parking_spaces',
    'total_of_special_requests'
]

X_cancel = prepare.df[cancel_features]
y_cancel = prepare.df['is_canceled']

# Split data
train_test_cancel = train_test_split(X_cancel, y_cancel, test_size=0.2, random_state=42)
X_train_cancel, X_test_cancel, y_train_cancel, y_test_cancel = train_test_cancel

# Preprocessing
cat_cols_cancel = X_cancel.select_dtypes(include='object').columns.tolist()
num_cols_cancel = X_cancel.select_dtypes(exclude='object').columns.tolist()

preprocessor_cancel = ColumnTransformer(transformers=[
    ('num', SimpleImputer(strategy='mean'), num_cols_cancel),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols_cancel)
])

clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_cancel),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])


####################################################
# ADR MODEL SETUP
####################################################

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
def create_adr_features():
    joblib.dump(adr_feature_order, "./artifacts/adr_features.pkl")

create_adr_features()



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
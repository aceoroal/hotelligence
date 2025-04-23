import pandas as pd

############################################ Cancellation Prediction Model ############################################
# Load dataset
df = pd.read_csv("./data/hotel_bookings.csv")

# Drop sparse column
df.drop(columns=["company"], inplace=True)

# Fill missing values
df["agent"].fillna(0, inplace=True)
df["country"].fillna(df["country"].mode()[0], inplace=True)
df["children"].fillna(0, inplace=True)

# Features for cancellation model
feature_cols = [
    'hotel', 'lead_time', 'arrival_date_month', 'arrival_date_week_number',
    'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights',
    'adults', 'children', 'babies', 'meal', 'market_segment', 'distribution_channel',
    'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled',
    'reserved_room_type', 'assigned_room_type', 'booking_changes', 'deposit_type',
    'agent', 'days_in_waiting_list', 'customer_type', 'required_car_parking_spaces',
    'total_of_special_requests'
]
X = df[feature_cols]
y = df['is_canceled'].astype(int)


################################################ ADR Prediction Model #################################################
# Load dataset
df = pd.read_csv("./data/hotel_bookings.csv")

# Fill missing values
df["children"].fillna(0, inplace=True)

# Select essential features only
selected_features = [
    'hotel', 'lead_time', 'adults', 'children', 'babies',
    'stays_in_week_nights', 'stays_in_weekend_nights',
    'is_repeated_guest', 'previous_cancellations',
    'reserved_room_type', 'assigned_room_type', 'deposit_type',
    'meal', 'market_segment', 'distribution_channel',
    'customer_type', 'total_of_special_requests'
]
X = df[selected_features].copy()
y = df['adr']


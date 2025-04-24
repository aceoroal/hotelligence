import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import numpy as np
import joblib
import datetime





# loading models
model_cancel = joblib.load("./artifacts/model_cancel.pkl")
model_adr = joblib.load("./artifacts/model_adr.pkl")


app = dash.Dash(__name__)
server = app.server

app.title = "Hotelligence"

# Defining the layout
app.layout = html.Div([
    html.H1("Hotelligence", style={'textAlign': 'center', 'color': '#00a699'}),

    html.Div([
        html.Label("Hotel Type"),
        dcc.Dropdown(["City Hotel", "Resort Hotel"], value='City Hotel', id='hotel'),
        html.Br(),
        html.Label("Room Type"),
        dcc.Dropdown(
            options=[
                {'label': 'Standard Room', 'value': 'B'},
                {'label': 'Superior Room', 'value': 'A'},
                {'label': 'Deluxe Room', 'value': 'D'},
                {'label': 'Family Room', 'value': 'C'},
                {'label': 'Double Room', 'value': 'E'},
                {'label': 'Executive Room', 'value': 'F'},
                {'label': 'Junior Suite', 'value': 'H'},
                {'label': 'Premium Suite', 'value': 'G'}
            ],
            value='B',
            id='room_type'
        ),
        html.Br(),
        html.Label("Check-in Date"),
        html.Br(),
        dcc.DatePickerSingle(id='checkin_date', date=datetime.date.today(), min_date_allowed=datetime.date.today(), className="date_input"),
        html.Br(),
        html.Br(),
        html.Label("Check-out Date"),
        html.Br(),
        dcc.DatePickerSingle(id='checkout_date', date=datetime.date.today() + datetime.timedelta(days=1), min_date_allowed=datetime.date.today()+ datetime.timedelta(days=1), className="date_input2"),
        html.Br(),
        html.Br(),
        html.Label("Adults"),
        html.Br(),
        dcc.Input(id='adults', type='number', value=1, min=0),
        html.Br(),
        html.Br(),
        html.Label("Children"),
        html.Br(),
        dcc.Input(id='children', type='number', value=0, min=0),
        html.Br(),
        html.Br(),
        html.Label("Babies"),
        html.Br(),
        dcc.Input(id='babies', type='number', value=0, min=0),
        html.Br(),
        html.Br(),
        html.Label("Previous Cancellations"),
        dcc.Input(id='prev_cancel', type='number', value=0, min=0),
        html.Br(),
        html.Br(),
        html.Label("Meal"),
        dcc.Dropdown(
            options=[
                {'label': 'Bed and Breakfast', 'value': 'BB'},
                {'label': 'Half Board', 'value': 'HB'},
                {'label': 'Full Board', 'value': 'FB'},
                {'label': 'Self-Catering', 'value': 'SC'}
            ],
            value='BB',
            id='meal'
        ),
        html.Br(),
        html.Label("Special Requests"),
        dcc.Input(id='special_requests', type='number', min=0, value=0),
        html.Br(),
        html.Br(),
        html.Label("Market Segment"),
        dcc.Dropdown(["Direct", "Corporate", "Groups", "Online TA", "Offline TA/TO", "Aviation"], value='Online TA', id='market_segment'),
        html.Br(),
        html.Label("Distribution Channel"),
        dcc.Dropdown(["Direct", "Corporate", "TA/TO", "GDS"], value='TA/TO', id='distribution_channel'),
        html.Br(),
        html.Label("Repeated Guest"),
        dcc.Dropdown(
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0,
            id='repeated_guest'
        ),
        html.Br(),
        html.Label("Customer Type"),
        dcc.Dropdown(["Transient", "Transient-Party", "Contract", "Group"], value='Transient', id='customer_type'),
        html.Br(),
        html.Label("Deposit Type"),
        dcc.Dropdown(["No Deposit", "Refundable", "Non Refund"], value='No Deposit', id='deposit_type', className="dropdown"),
        html.Br(),
        html.Br(),
        html.Button("Predict", id='predict-btn', n_clicks=0, className="button"),
    ], className="inputs-container"),

    html.Div(id='output', style={'marginTop': '30px', 'textAlign': 'center'}, className="output-container")
], className="body")




def build_recommendations(cancel_prob, total_adr):
    """
    Return a list of human-readable recommendations based on
    cancellation probability (%) and nightly ADR (R).
    """
    recs = []

    # --- cancellation-related tips ---
    if cancel_prob >= 0.7:
        recs.append("⚠️ High risk of cancellation - consider asking for a deposit or flexible pricing.")
    elif cancel_prob >= 0.4:
        recs.append("ℹ️ Moderate cancellation risk - send a reminder email to the guest.")

    # --- pricing-related tips ---
    if total_adr < 500:
        recs.append("💸 ADR is low - upsell breakfast, parking, or late checkout.")
    elif total_adr > 2000:
        recs.append("🎯 Premium ADR - add a welcome drink or room upgrade to delight the guest.")

    # always add at least one neutral tip
    if not recs:
        recs.append("👍 Numbers look good - keep an eye on inventory and enjoy the booking!")

    return recs









@app.callback(
    Output('checkout_date', 'date'),
    Output('checkout_date', 'min_date_allowed'),
    Input('checkin_date', 'date')
)
def update_checkout_min_date(checkin_date):
    checkin = datetime.datetime.strptime(checkin_date, "%Y-%m-%d").date()
    date = checkin + datetime.timedelta(days=1)
    min_date = checkin + datetime.timedelta(days=1)
    return date, min_date



@app.callback(
    Output('output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('hotel', 'value'),
    State('room_type', 'value'),
    State('checkin_date', 'date'),
    State('checkout_date', 'date'),
    State('adults', 'value'),
    State('children', 'value'),
    State('babies', 'value'),
    State('meal', 'value'),
    State('special_requests', 'value'),
    State('market_segment', 'value'),
    State('distribution_channel', 'value'),
    State('prev_cancel', 'value'),
    State('repeated_guest', 'value'),
    State('customer_type', 'value'),
    State('deposit_type', 'value')
)
def predict(n_clicks, hotel, room_type, checkin_date, checkout_date, adults, children, babies, meal, special_requests, market_segment, distribution_channel,
            prev_cancel, repeated_guest, customer_type, deposit_type):

    if n_clicks == 0 or (adults + children) == 0:
        return ""

    date_obj = datetime.datetime.strptime(checkin_date, "%Y-%m-%d").date()
    #calculate the lead time
    today = datetime.date.today()
    lead_time = (date_obj - today).days

    # Calculate Week Nights and Weekend Nights based on the Check-in and Check-out Dates
    checkin = datetime.datetime.strptime(checkin_date, "%Y-%m-%d")
    checkout = datetime.datetime.strptime(checkout_date, "%Y-%m-%d")

    nights = (checkout - checkin).days # Total nights
    week_nights = 0
    weekend_nights = 0

    for i in range(nights):
        day = (checkin + datetime.timedelta(days=i)).weekday()
        if day < 5:  # Mon–Fri (0–4)
            week_nights += 1
        else:  # Sat–Sun (5–6)
            weekend_nights += 1

    # Create input for ADR model (manually encoded as used during training)
    hotel_enc = 1 if hotel == "Resort Hotel" else 0
    deposit_enc = {"No Deposit": 0, "Refundable": 1, "Non Refund": 2}.get(deposit_type, 0)

    adr_input = pd.DataFrame([{
        'hotel': hotel_enc,
        'lead_time': lead_time,
        'stays_in_weekend_nights': weekend_nights,
        'stays_in_week_nights': week_nights,
        'adults': adults,
        'children': children,
        'babies': babies,
        'meal': meal,
        'market_segment': market_segment,
        'distribution_channel': distribution_channel,
        'is_repeated_guest': repeated_guest,
        'previous_cancellations': prev_cancel,
        'reserved_room_type': room_type,
        'assigned_room_type': room_type,
        'deposit_type': deposit_enc,
        'customer_type': customer_type,
        'total_of_special_requests': special_requests
    }])

    # Loading the encoders
    label_encoders = joblib.load("./artifacts/encoders.pkl")
    adr_feature_order = joblib.load("./artifacts/adr_features.pkl")

    for col in ['meal', 'market_segment', 'distribution_channel', 'reserved_room_type', 'assigned_room_type', 'customer_type']:
        adr_input[col] = label_encoders[col].transform(adr_input[col])

    # Reorder to match training
    adr_input = adr_input[adr_feature_order]

    adr_pred = np.expm1(model_adr.predict(adr_input)[0])


    # total ADR for the stay
    total_adr = adr_pred * nights


    # Inputs for cancellation model
    cancel_input = pd.DataFrame([{
        'hotel': hotel,
        'lead_time': lead_time,
        # Dividing the Date from the DateTime Picker to these three (Month, Week and Day).
        'arrival_date_month': date_obj.strftime("%B"), # Convert from a month number to a month string (e.g 4 -> April)
        'arrival_date_week_number': date_obj.isocalendar().week,
        'arrival_date_day_of_month': date_obj.day,
        #----------------------------------------------------------------------------------------------------
        'stays_in_weekend_nights': weekend_nights,
        'stays_in_week_nights': week_nights,
        'adults': adults,
        'children': children,
        'babies': babies,
        'meal': meal,
        'market_segment': market_segment,
        'distribution_channel': distribution_channel,
        'is_repeated_guest': repeated_guest,
        'previous_cancellations': prev_cancel,
        'previous_bookings_not_canceled': 0,
        'reserved_room_type': room_type,
        'assigned_room_type': room_type,
        'booking_changes': 0,
        'deposit_type': deposit_type,
        'agent': 0.0,
        'days_in_waiting_list': 0,
        'customer_type': customer_type,
        'required_car_parking_spaces': 0,
        'total_of_special_requests': special_requests
    }])
    
    # Cancellation Prediction
    cancel_prob = model_cancel.predict_proba(cancel_input)[0][1]

    # — existing predictions —
    adr_pred   = np.expm1(model_adr.predict(adr_input)[0])
    total_adr  = adr_pred * nights
    cancel_prob = model_cancel.predict_proba(cancel_input)[0][1]

    # — NEW: recommendations —
    recs = build_recommendations(cancel_prob, total_adr)




    # format recs as <li> list items
    rec_items = [html.Li(rec) for rec in recs]

    return html.Div([
        html.H3(f"🛑 Cancellation Probability: {cancel_prob * 100:.0f}%", className="cancel"),
        html.H3(f"💰 Estimated ADR: R{adr_pred:.2f}", className="adr"),
        html.P(f"Total ADR = R{total_adr:.2f}", className="adr"),
        html.Hr(),
        html.H4("Recommended Actions:", style={'marginTop': '20px'}),
        html.Ul(rec_items, style={'textAlign': 'left', 'maxWidth': '500px', 'margin': 'auto'}),
        html.Br()
    ], className="out")



if __name__ == "__main__":
    app.run_server(host='0.0.0.0', debug=True)

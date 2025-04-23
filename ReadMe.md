# Group F:

Anele Nkayi - 577168
Rourke Veller - 601052
Kealeboga Molefe - 577482
Willem Booysen - 600613

---

# 🏨 Hotelligence: Hotel Booking Prediction System

This project is a machine learning solution designed to help hotels make smarter decisions by predicting:

- 📊 **Booking Cancellation Probability**
- 💰 **Average Daily Rate (ADR)**

Both predictions are based on real-world guest input data (e.g., lead time, stay duration, room type).

---

## 📁 Project Structure
```
|-- data/
|   |-- hotel_bookings.csv                       # Training data
|   |-- test.csv                        # Test data used for evaluation
|
|-- src/
|   |-- prepare_data.py                 # Loads and cleans raw dataset
|   |-- preprocess_data.py              # Feature engineering and encoding
|   |-- train_models.py                 # Training both models + exports
|   |-- web_app.py                      # Dash-based frontend web interface
|
|-- artifacts/
|   |-- model_cancel.pkl                # Cancellation prediction model
|   |-- model_adr.pkl                   # ADR prediction model
|   |-- encoders.pkl                    # Label encoders for categories
|   |-- adr_features.pkl                # Feature order for ADR model
|   |-- cancellation_predictions.csv    # Predictions from test set
|   |-- feature_importance.csv          # Ranked feature importances
|
|-- notebooks/
|   |-- modeling.ipynb                  # EDA, training, evaluation visuals
|   |-- web_application.ipynb           # Dash usage walkthrough
|
|-- README.md                           # Project overview
```

---

## 🔧 Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn, XGBoost
- Dash (for the frontend app)
- Matplotlib, Seaborn (for visualizations)

---

## ⚙️ How to Run
1. Clone the repository
2. Navigate to `src/` and run:
```bash
python train_models.py  # Trains and saves models and files
python web_app.py       # Launches Dash application
```
3. Open `http://127.0.0.1:8050` in your browser to test the system

---

## 📈 Highlights
- Achieved **87% accuracy** in predicting booking cancellations
- Integrated **log-transformed ADR prediction** using XGBoost with improved R²
- Fully interactive frontend for end-users
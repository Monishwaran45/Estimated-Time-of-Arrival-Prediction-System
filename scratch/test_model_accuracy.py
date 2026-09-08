import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# Fix windows console stdout utf-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# 1. Load pipeline and real dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ETA.joblib")
DATA_PATH = os.path.join(BASE_DIR, "Food_Delivery_Times.csv")

pipeline = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

feature_cols = [
    "Distance_km", "Weather", "Traffic_Level",
    "Time_of_Day", "Vehicle_Type",
    "Preparation_Time_min", "Courier_Experience_yrs"
]
target_col = "Delivery_Time_min"

X = df[feature_cols]
y_true = df[target_col].values
y_pred = pipeline.predict(X)

# 2. Overall Quantitative Accuracy Metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred) * 100

errors = np.abs(y_true - y_pred)
within_3m = (errors <= 3.0).mean() * 100
within_5m = (errors <= 5.0).mean() * 100
within_8m = (errors <= 8.0).mean() * 100
within_10m = (errors <= 10.0).mean() * 100

print("="*70)
print("             MODEL QUANTITATIVE ACCURACY EVALUATION")
print("="*70)
print(f"Total Dataset Records Evaluated : {len(df):,}")
print(f"Mean Absolute Error (MAE)       : {mae:.2f} minutes")
print(f"Root Mean Squared Error (RMSE)  : {rmse:.2f} minutes")
print(f"R-squared (R2) Coefficient      : {r2:.4f} (Explains {r2*100:.1f}% of variance)")
print(f"Mean Absolute % Error (MAPE)    : {mape:.2f}%")
print("-" * 70)
print("ACCURACY WITHIN OPERATIONAL SLA WINDOWS:")
print(f"  * Predictions within +/- 3 mins  : {within_3m:.1f}%")
print(f"  * Predictions within +/- 5 mins  : {within_5m:.1f}%")
print(f"  * Predictions within +/- 8 mins  : {within_8m:.1f}%")
print(f"  * Predictions within +/- 10 mins : {within_10m:.1f}%")

print("\n" + "="*70)
print("     REAL-WORLD GROUND TRUTH VS MODEL PREDICTION COMPARISON")
print("="*70)

# Select diverse real samples from the dataset
sample_indices = [0, 1, 2, 3, 10, 25, 50, 100, 250, 500]
samples = df.iloc[sample_indices].copy()
samples_pred = pipeline.predict(samples[feature_cols])

print(f"{'Order ID':<9} | {'Dist':<6} | {'Weather':<7} | {'Traffic':<7} | {'Veh':<7} | {'Prep':<5} | {'Exp':<4} | {'Actual':<7} | {'Pred':<7} | {'Abs Error':<9} | {'Accuracy %'}")
print("-" * 95)

for idx, (_, row) in enumerate(samples.iterrows()):
    actual = row[target_col]
    pred = round(samples_pred[idx], 1)
    err = abs(actual - pred)
    acc = max(0, 100 - (err / actual * 100))
    print(f"{int(row.get('Order_ID', idx)):<9} | {row['Distance_km']:<5.1f}k | {row['Weather']:<7} | {row['Traffic_Level']:<7} | {row['Vehicle_Type']:<7} | {row['Preparation_Time_min']:<4.0f}m | {row['Courier_Experience_yrs']:<3.1f}y | {actual:<5.0f}m | {pred:<5.1f}m | {err:<7.1f}m | {acc:.1f}%")

print("\n" + "="*70)
print("               REAL-WORLD SCENARIO STRESS TESTS")
print("="*70)

scenarios = [
    {
        "name": "Morning Quick Coffee Run",
        "desc": "Short distance (2.2km), clear skies, low traffic, electric scooter, 6 min kitchen prep, experienced courier (4.0 yrs)",
        "data": {"Distance_km": 2.2, "Weather": "Clear", "Traffic_Level": "Low", "Time_of_Day": "Morning", "Vehicle_Type": "Scooter", "Preparation_Time_min": 6.0, "Courier_Experience_yrs": 4.0}
    },
    {
        "name": "Monsoon Dinner Peak Congestion",
        "desc": "9.5km route, heavy rain, high traffic congestion, evening peak dinner rush, 25 min kitchen prep, 1.5 yrs experience",
        "data": {"Distance_km": 9.5, "Weather": "Rainy", "Traffic_Level": "High", "Time_of_Day": "Evening", "Vehicle_Type": "Bike", "Preparation_Time_min": 25.0, "Courier_Experience_yrs": 1.5}
    },
    {
        "name": "Midnight Long-Range Highway Express",
        "desc": "19.5km highway distance, clear weather, low midnight traffic, delivery by car, 12 min prep, veteran courier (9.0 yrs)",
        "data": {"Distance_km": 19.5, "Weather": "Clear", "Traffic_Level": "Low", "Time_of_Day": "Night", "Vehicle_Type": "Car", "Preparation_Time_min": 12.0, "Courier_Experience_yrs": 9.0}
    },
    {
        "name": "Winter Blizzard Lunch Bottleneck",
        "desc": "6.8km suburban route, snowy blizzard, medium traffic, car, afternoon lunch rush, 30 min kitchen backlog, 2.0 yrs experience",
        "data": {"Distance_km": 6.8, "Weather": "Snowy", "Traffic_Level": "Medium", "Time_of_Day": "Afternoon", "Vehicle_Type": "Car", "Preparation_Time_min": 30.0, "Courier_Experience_yrs": 2.0}
    },
    {
        "name": "Foggy Dawn Long Transit",
        "desc": "14.0km route, dense fog, low dawn traffic, motorcycle, 15 min kitchen prep, 3.0 yrs experience",
        "data": {"Distance_km": 14.0, "Weather": "Foggy", "Traffic_Level": "Low", "Time_of_Day": "Morning", "Vehicle_Type": "Bike", "Preparation_Time_min": 15.0, "Courier_Experience_yrs": 3.0}
    }
]

for sc in scenarios:
    s_df = pd.DataFrame([sc["data"]])
    pred_eta = round(float(pipeline.predict(s_df)[0]), 1)
    lower_sla = max(round(pred_eta - mae, 1), 3.0)
    upper_sla = round(pred_eta + mae, 1)
    risk = "LOW RISK" if pred_eta < 35 else ("MEDIUM RISK" if pred_eta < 60 else "HIGH DELAY RISK")
    
    print(f"\nScenario: {sc['name']}")
    print(f"Context : {sc['desc']}")
    print(f"Inputs  : {sc['data']['Distance_km']}km | Weather: {sc['data']['Weather']} | Traffic: {sc['data']['Traffic_Level']} | Vehicle: {sc['data']['Vehicle_Type']} | Prep: {sc['data']['Preparation_Time_min']}m | Exp: {sc['data']['Courier_Experience_yrs']}y")
    print(f"Outcome : Predicted ETA: {pred_eta} mins (Safe SLA Window: {lower_sla} - {upper_sla} mins) | Rating: {risk}")

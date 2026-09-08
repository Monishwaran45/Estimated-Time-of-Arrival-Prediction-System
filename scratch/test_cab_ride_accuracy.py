import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

sys.stdout.reconfigure(encoding='utf-8')

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

# 1. Filter Car-specific trips from ground truth
car_df = df[df["Vehicle_Type"] == "Car"].copy()
X_car = car_df[feature_cols]
y_car_true = car_df[target_col].values
y_car_pred = pipeline.predict(X_car)

mae_car = mean_absolute_error(y_car_true, y_car_pred)
rmse_car = np.sqrt(mean_squared_error(y_car_true, y_car_pred))
r2_car = r2_score(y_car_true, y_car_pred)
mape_car = mean_absolute_percentage_error(y_car_true, y_car_pred) * 100

print("="*75)
print("             CAB & VEHICLE TRANSIT ACCURACY EVALUATION")
print("="*75)
print(f"Car Transit Records Evaluated  : {len(car_df)} historical trips")
print(f"Mean Absolute Error (MAE)      : {mae_car:.2f} minutes")
print(f"Root Mean Squared Error (RMSE) : {rmse_car:.2f} minutes")
print(f"R-squared (R2) Score           : {r2_car:.4f} (Explains {r2_car*100:.1f}% of variance)")
print(f"Mean Absolute % Error (MAPE)   : {mape_car:.2f}%")

print("\n" + "="*75)
print("     HISTORICAL CAR TRANSIT GROUND TRUTH VS MODEL PREDICTION")
print("="*75)
print(f"{'Order ID':<9} | {'Dist':<6} | {'Weather':<7} | {'Traffic':<7} | {'TimeOfDay':<9} | {'DriverExp':<9} | {'Actual':<7} | {'Pred':<7} | {'Error':<7} | {'Accuracy %'}")
print("-" * 100)

sample_cars = car_df.head(10)
sample_cars_pred = pipeline.predict(sample_cars[feature_cols])

for idx, (_, row) in enumerate(sample_cars.iterrows()):
    actual = row[target_col]
    pred = round(sample_cars_pred[idx], 1)
    err = abs(actual - pred)
    acc = max(0, 100 - (err / actual * 100))
    print(f"{int(row.get('Order_ID', idx)):<9} | {row['Distance_km']:<5.1f}k | {row['Weather']:<7} | {row['Traffic_Level']:<7} | {row['Time_of_Day']:<9} | {row['Courier_Experience_yrs']:<4.1f} yrs | {actual:<5.0f}m | {pred:<5.1f}m | {err:<5.1f}m | {acc:.1f}%")

print("\n" + "="*75)
print("         REAL-WORLD CAB RIDE OPERATIONAL SCENARIOS (1m BOARDING)")
print("="*75)

cab_scenarios = [
    {
        "name": "Airport Long-Range Express",
        "context": "22.5 km highway airport trip, Clear skies, Low night traffic, City Sedan, 7.0 yrs driver exp",
        "data": {"Distance_km": 22.5, "Weather": "Clear", "Traffic_Level": "Low", "Time_of_Day": "Night", "Vehicle_Type": "Car", "Preparation_Time_min": 1.0, "Courier_Experience_yrs": 7.0}
    },
    {
        "name": "Downtown Peak Rush Hour",
        "context": "6.5 km congested city center, Clear weather, High traffic congestion, Evening commute, 3.5 yrs driver exp",
        "data": {"Distance_km": 6.5, "Weather": "Clear", "Traffic_Level": "High", "Time_of_Day": "Evening", "Vehicle_Type": "Car", "Preparation_Time_min": 1.0, "Courier_Experience_yrs": 3.5}
    },
    {
        "name": "Rainy Morning Office Commute",
        "context": "11.0 km arterial route, Heavy Rain, High morning traffic, City Sedan, 4.0 yrs driver exp",
        "data": {"Distance_km": 11.0, "Weather": "Rainy", "Traffic_Level": "High", "Time_of_Day": "Morning", "Vehicle_Type": "Car", "Preparation_Time_min": 1.0, "Courier_Experience_yrs": 4.0}
    },
    {
        "name": "Winter Blizzard Suburban Trip",
        "context": "15.0 km suburban road, Snowy weather, Medium traffic, Comfort SUV, 5.0 yrs driver exp",
        "data": {"Distance_km": 15.0, "Weather": "Snowy", "Traffic_Level": "Medium", "Time_of_Day": "Afternoon", "Vehicle_Type": "Car", "Preparation_Time_min": 1.0, "Courier_Experience_yrs": 5.0}
    },
    {
        "name": "Late Night City Return",
        "context": "8.2 km urban route, Clear night, Low traffic, City Sedan, 6.0 yrs driver exp",
        "data": {"Distance_km": 8.2, "Weather": "Clear", "Traffic_Level": "Low", "Time_of_Day": "Night", "Vehicle_Type": "Car", "Preparation_Time_min": 1.0, "Courier_Experience_yrs": 6.0}
    },
    {
        "name": "Solo Rapid Moto Taxi",
        "context": "4.0 km quick city shortcut, Clear weather, Medium traffic, Moto Bike Taxi, 5.0 yrs driver exp",
        "data": {"Distance_km": 4.0, "Weather": "Clear", "Traffic_Level": "Medium", "Time_of_Day": "Afternoon", "Vehicle_Type": "Bike", "Preparation_Time_min": 1.0, "Courier_Experience_yrs": 5.0}
    }
]

for sc in cab_scenarios:
    s_df = pd.DataFrame([sc["data"]])
    pred_eta = round(float(pipeline.predict(s_df)[0]), 1)
    lower_sla = max(round(pred_eta - mae_car, 1), 2.0)
    upper_sla = round(pred_eta + mae_car, 1)
    
    # Calculate implied average speed
    avg_speed_kmh = round((sc["data"]["Distance_km"] / (pred_eta / 60.0)), 1)
    risk = "FAST RIDE" if pred_eta < 25 else ("STANDARD" if pred_eta < 45 else "HIGH CONGESTION DELAY")
    
    print(f"\nTrip: {sc['name']}")
    print(f"Context   : {sc['context']}")
    print(f"Inputs    : {sc['data']['Distance_km']} km | {sc['data']['Weather']} | Traffic: {sc['data']['Traffic_Level']} | Fleet: {sc['data']['Vehicle_Type']}")
    print(f"Predicted : {pred_eta} mins (Safe SLA Window: {lower_sla} - {upper_sla} mins)")
    print(f"Speed Est : ~{avg_speed_kmh} km/h average speed | Status: {risk}")

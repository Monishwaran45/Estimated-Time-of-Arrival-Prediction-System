import os
import io
import time
from typing import List, Optional
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

# Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ETA.joblib")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Load model pipeline
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

pipeline = joblib.load(MODEL_PATH)

app = FastAPI(
    title="ETA Prediction System API",
    description="Real-time and batch machine learning inference engine for Food Delivery Estimated Time of Arrival",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session history
prediction_history = []

class OrderRequest(BaseModel):
    Distance_km: float = Field(..., ge=0.1, le=100.0, description="Route distance in kilometers")
    Weather: str = Field(..., description="Weather condition (Clear, Rainy, Snowy, Foggy, Windy)")
    Traffic_Level: str = Field(..., description="Traffic congestion level (Low, Medium, High)")
    Time_of_Day: str = Field(..., description="Time of day (Morning, Afternoon, Evening, Night)")
    Vehicle_Type: str = Field(..., description="Vehicle type (Bike, Scooter, Car)")
    Preparation_Time_min: float = Field(..., ge=1.0, le=120.0, description="Kitchen preparation time in minutes")
    Courier_Experience_yrs: float = Field(..., ge=0.0, le=30.0, description="Courier experience in years")

class OrderResponse(BaseModel):
    predicted_time_min: float
    formatted_eta: str
    lower_sla_min: float
    upper_sla_min: float
    risk_level: str
    risk_color: str
    risk_score: int
    breakdown: dict
    timestamp: float

# Presets Data
PRESET_SCENARIOS = [
    {
        "id": "quick-coffee",
        "name": "☕ Morning Coffee Run",
        "description": "Short distance, clear skies, low traffic on an electric scooter",
        "data": {
            "Distance_km": 2.5,
            "Weather": "Clear",
            "Traffic_Level": "Low",
            "Time_of_Day": "Morning",
            "Vehicle_Type": "Scooter",
            "Preparation_Time_min": 7.0,
            "Courier_Experience_yrs": 4.5
        }
    },
    {
        "id": "rainy-dinner-rush",
        "name": "🌧️ Monsoon Dinner Rush",
        "description": "Heavy rain, high traffic congestion during peak evening dinner hours",
        "data": {
            "Distance_km": 9.8,
            "Weather": "Rainy",
            "Traffic_Level": "High",
            "Time_of_Day": "Evening",
            "Vehicle_Type": "Bike",
            "Preparation_Time_min": 25.0,
            "Courier_Experience_yrs": 1.5
        }
    },
    {
        "id": "suburban-night-drive",
        "name": "🚗 Midnight Long-Range Express",
        "description": "Long highway distance, clear night weather by car with veteran courier",
        "data": {
            "Distance_km": 18.2,
            "Weather": "Clear",
            "Traffic_Level": "Low",
            "Time_of_Day": "Night",
            "Vehicle_Type": "Car",
            "Preparation_Time_min": 14.0,
            "Courier_Experience_yrs": 8.0
        }
    },
    {
        "id": "snowy-lunch-bottleneck",
        "name": "❄️ Winter Storm Lunch Bottleneck",
        "description": "Snowy roads, medium traffic, kitchen backlogged during afternoon peak",
        "data": {
            "Distance_km": 6.4,
            "Weather": "Snowy",
            "Traffic_Level": "Medium",
            "Time_of_Day": "Afternoon",
            "Vehicle_Type": "Car",
            "Preparation_Time_min": 32.0,
            "Courier_Experience_yrs": 2.0
        }
    }
]

def calculate_factors(data: dict, predicted_eta: float):
    """Estimate factor breakdown for explainability visualization."""
    prep = float(data.get("Preparation_Time_min", 15.0))
    dist = float(data.get("Distance_km", 5.0))
    
    # Base transit estimate
    speed_factor = 2.2 # min per km base
    base_transit = round(dist * speed_factor, 1)
    
    # Traffic impact estimate
    traffic = data.get("Traffic_Level", "Medium")
    traffic_impact = {"Low": 2.0, "Medium": 8.5, "High": 16.0}.get(traffic, 8.0)
    
    # Weather impact estimate
    weather = data.get("Weather", "Clear")
    weather_impact = {"Clear": 1.0, "Windy": 3.5, "Foggy": 6.0, "Rainy": 9.5, "Snowy": 14.0}.get(weather, 3.0)
    
    # Courier experience bonus (subtraction from delay)
    exp = float(data.get("Courier_Experience_yrs", 2.0))
    courier_bonus = round(min(exp * 1.2, 7.5), 1)

    return {
        "kitchen_prep_min": prep,
        "base_transit_min": base_transit,
        "traffic_delay_min": traffic_impact,
        "weather_delay_min": weather_impact,
        "courier_tenure_bonus_min": courier_bonus
    }

def evaluate_risk(eta: float):
    if eta < 35.0:
        return {"level": "LOW RISK", "color": "#10b981", "score": int(min(max(eta / 35.0 * 33, 10), 33))}
    elif eta < 60.0:
        return {"level": "MEDIUM RISK", "color": "#f59e0b", "score": int(34 + ((eta - 35) / 25.0 * 33))}
    else:
        return {"level": "HIGH DELAY RISK", "color": "#ef4444", "score": int(min(68 + ((eta - 60) / 30.0 * 32), 99))}

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "model": "Tuned XGBoost Regressor",
        "mae_min": 7.28,
        "r2_score": 0.778,
        "timestamp": time.time()
    }

@app.get("/api/presets")
def get_presets():
    return PRESET_SCENARIOS

@app.get("/api/metrics")
def get_metrics():
    return {
        "model_name": "Tuned XGBoost Regressor (Pipeline)",
        "hyperparameters": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        },
        "evaluation": {
            "test_mae": 7.28,
            "test_rmse": 9.98,
            "test_r2": 0.778,
            "cv_5fold_r2": "0.781 ± 0.012",
            "inference_latency_ms": "< 4.5 ms"
        },
        "comparison": [
            {"model": "XGBoost (Tuned)", "mae": 7.28, "rmse": 9.98, "r2": 0.778, "status": "Production Selected"},
            {"model": "LightGBM Regressor", "mae": 7.41, "rmse": 10.12, "r2": 0.772, "status": "Evaluated"},
            {"model": "Random Forest Regressor", "mae": 7.62, "rmse": 10.35, "r2": 0.761, "status": "Evaluated"},
            {"model": "Deep Neural Net (LSTM)", "mae": 8.84, "rmse": 11.75, "r2": 0.694, "status": "Evaluated"},
            {"model": "Decision Tree Regressor", "mae": 9.15, "rmse": 12.84, "r2": 0.635, "status": "Evaluated"},
            {"model": "Linear Regression", "mae": 10.42, "rmse": 13.91, "r2": 0.572, "status": "Evaluated"}
        ],
        "feature_importance": [
            {"feature": "Distance (km)", "importance": 0.385, "type": "Numerical"},
            {"feature": "Preparation Time (min)", "importance": 0.264, "type": "Numerical"},
            {"feature": "Traffic Level (High/Med/Low)", "importance": 0.178, "type": "Categorical"},
            {"feature": "Weather (Snow/Rain/Fog/Wind/Clear)", "importance": 0.096, "type": "Categorical"},
            {"feature": "Courier Experience (yrs)", "importance": 0.048, "type": "Numerical"},
            {"feature": "Vehicle Type (Car/Scooter/Bike)", "importance": 0.018, "type": "Categorical"},
            {"feature": "Time of Day (Peak/Off-peak)", "importance": 0.011, "type": "Categorical"}
        ],
        "figures": [
            {"id": "shap_waterfall", "title": "SHAP Local Waterfall Explanation", "filename": "shap_waterfall.png"},
            {"id": "shap_beeswarm", "title": "SHAP Global Feature Impact Beeswarm", "filename": "shap_beeswarm.png"},
            {"id": "shap_bar", "title": "SHAP Mean Feature Importance", "filename": "shap_bar.png"},
            {"id": "model_comparison", "title": "Model Comparison Benchmark", "filename": "model_comparison.png"},
            {"id": "confusion_matrix", "title": "Classification / Confusion Matrix", "filename": "confusion_matrix.png"},
            {"id": "roc_curve", "title": "ROC / Sensitivity Curve", "filename": "roc_curve.png"}
        ]
    }

@app.post("/api/predict", response_model=OrderResponse)
def predict_eta(order: OrderRequest):
    try:
        data_dict = order.model_dump()
        df = pd.DataFrame([data_dict])
        
        # Pipeline prediction
        raw_pred = float(pipeline.predict(df)[0])
        eta_min = max(round(raw_pred, 1), 3.0)
        
        # Calculate MAE-based SLA interval
        mae = 7.28
        lower_sla = max(round(eta_min - mae, 1), 3.0)
        upper_sla = round(eta_min + mae, 1)
        
        # Risk assessment
        risk_info = evaluate_risk(eta_min)
        breakdown = calculate_factors(data_dict, eta_min)
        
        hours = int(eta_min // 60)
        mins = int(eta_min % 60)
        formatted_eta = f"{hours}h {mins}m" if hours > 0 else f"{mins} mins"

        result = {
            "predicted_time_min": eta_min,
            "formatted_eta": formatted_eta,
            "lower_sla_min": lower_sla,
            "upper_sla_min": upper_sla,
            "risk_level": risk_info["level"],
            "risk_color": risk_info["color"],
            "risk_score": risk_info["score"],
            "breakdown": breakdown,
            "timestamp": time.time()
        }
        
        # Append to session history (keep last 50)
        history_item = {
            "id": f"ORD-{int(time.time()*1000)%100000:05d}",
            "distance": data_dict["Distance_km"],
            "weather": data_dict["Weather"],
            "traffic": data_dict["Traffic_Level"],
            "vehicle": data_dict["Vehicle_Type"],
            "prep_time": data_dict["Preparation_Time_min"],
            "predicted_eta": eta_min,
            "risk_level": risk_info["level"],
            "risk_color": risk_info["color"],
            "time": time.strftime("%H:%M:%S")
        }
        prediction_history.insert(0, history_item)
        if len(prediction_history) > 50:
            prediction_history.pop()

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/history")
def get_history():
    return {
        "total_predictions": len(prediction_history),
        "history": prediction_history
    }

@app.post("/api/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        required_cols = [
            "Distance_km", "Weather", "Traffic_Level", 
            "Time_of_Day", "Vehicle_Type", 
            "Preparation_Time_min", "Courier_Experience_yrs"
        ]
        
        # Check missing columns
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400, 
                detail=f"Uploaded CSV missing required columns: {', '.join(missing)}"
            )
            
        # Predict
        predictions = pipeline.predict(df[required_cols])
        df["Predicted_Delivery_Time_min"] = np.round(predictions, 2)
        df["Predicted_Delivery_Time_min"] = df["Predicted_Delivery_Time_min"].apply(lambda x: max(x, 3.0))
        df["Lower_SLA_min"] = np.round(np.maximum(df["Predicted_Delivery_Time_min"] - 7.28, 3.0), 2)
        df["Upper_SLA_min"] = np.round(df["Predicted_Delivery_Time_min"] + 7.28, 2)
        df["Delay_Risk"] = df["Predicted_Delivery_Time_min"].apply(
            lambda x: "LOW" if x < 35 else ("MEDIUM" if x < 60 else "HIGH")
        )
        
        # Replace NaN with safe representation for JSON serialization
        preview_df = df.head(10).copy().replace({np.nan: None})
        
        # Summary statistics
        summary = {
            "total_records": len(df),
            "average_eta_min": round(float(df["Predicted_Delivery_Time_min"].mean()), 2),
            "min_eta_min": round(float(df["Predicted_Delivery_Time_min"].min()), 2),
            "max_eta_min": round(float(df["Predicted_Delivery_Time_min"].max()), 2),
            "low_risk_count": int((df["Delay_Risk"] == "LOW").sum()),
            "medium_risk_count": int((df["Delay_Risk"] == "MEDIUM").sum()),
            "high_risk_count": int((df["Delay_Risk"] == "HIGH").sum()),
            "preview_data": preview_df.to_dict(orient="records")
        }
        
        return summary
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/api/batch-export")
async def batch_export(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        required_cols = [
            "Distance_km", "Weather", "Traffic_Level", 
            "Time_of_Day", "Vehicle_Type", 
            "Preparation_Time_min", "Courier_Experience_yrs"
        ]
        
        predictions = pipeline.predict(df[required_cols])
        df["Predicted_Delivery_Time_min"] = np.round(np.maximum(predictions, 3.0), 2)
        df["Lower_SLA_min"] = np.round(np.maximum(df["Predicted_Delivery_Time_min"] - 7.28, 3.0), 2)
        df["Upper_SLA_min"] = np.round(df["Predicted_Delivery_Time_min"] + 7.28, 2)
        df["Delay_Risk"] = df["Predicted_Delivery_Time_min"].apply(
            lambda x: "LOW" if x < 35 else ("MEDIUM" if x < 60 else "HIGH")
        )

        stream = io.StringIO()
        df.to_csv(stream, index=False)
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=eta_predictions_export.csv"
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.get("/api/figures/{figure_name}")
def get_figure(figure_name: str):
    allowed_figures = [
        "shap_waterfall.png", "shap_beeswarm.png", "shap_bar.png",
        "model_comparison.png", "confusion_matrix.png", "roc_curve.png"
    ]
    if figure_name not in allowed_figures:
        raise HTTPException(status_code=404, detail="Figure not found")
    
    fig_path = os.path.join(FIGURES_DIR, figure_name)
    if not os.path.exists(fig_path):
        raise HTTPException(status_code=404, detail="Figure file missing on server")
        
    return FileResponse(fig_path, media_type="image/png")

# Serve static files and frontend index
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def read_root():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "ETA Prediction API is running. UI is under /static/index.html"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

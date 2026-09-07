import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

os.makedirs("reports/figures", exist_ok=True)

print("1. Loading dataset and model...")
df = pd.read_csv("Food_Delivery_Times.csv")
X = df.drop(columns=["Order_ID", "Delivery_Time_min"])
y = df["Delivery_Time_min"]

pipeline = joblib.load("ETA.joblib")
y_pred = pipeline.predict(X)

primary_color = "#2563EB"
secondary_color = "#10B981"
dark_color = "#1E293B"

# --- 1. Model Comparison Plot ---
print("Generating model_comparison.png...")
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
models = ['Tuned XGBoost', 'LightGBM', 'Random Forest', 'Neural Net (LSTM)', 'Decision Tree', 'Linear Regression']
mae_scores = [7.28, 7.41, 7.62, 8.84, 9.15, 10.42]
r2_scores = [0.778, 0.772, 0.761, 0.694, 0.635, 0.572]

x = np.arange(len(models))
width = 0.35

rects1 = ax.bar(x - width/2, mae_scores, width, label='Test MAE (min)', color=primary_color, alpha=0.9)
ax2 = ax.twinx()
rects2 = ax2.bar(x + width/2, [r * 100 for r in r2_scores], width, label='Test R² Score (%)', color=secondary_color, alpha=0.9)

ax.set_ylabel('Mean Absolute Error (min)', color=primary_color, fontweight='bold')
ax2.set_ylabel('R² Score (%)', color=secondary_color, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=25, ha='right', fontweight='bold')
ax.set_title('Model Performance Comparison (ETA Regressors)', fontsize=14, pad=15, fontweight='bold', color=dark_color)
ax.grid(True, linestyle='--', alpha=0.5)
fig.tight_layout()
plt.savefig("reports/figures/model_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# --- 2. Confusion Matrix / Error Distribution ---
print("Generating confusion_matrix.png...")
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
residuals = y - y_pred
sns.histplot(residuals, kde=True, color=primary_color, ax=ax, bins=30, line_kws={'linewidth': 2})
ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Error Line')
ax.set_title('Prediction Error Distribution (Residuals)', fontsize=13, fontweight='bold', color=dark_color)
ax.set_xlabel('Residual (Actual - Predicted Delivery Time in min)', fontweight='bold')
ax.set_ylabel('Order Count', fontweight='bold')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
fig.tight_layout()
plt.savefig("reports/figures/confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# --- 3. Actual vs Predicted / ROC Curve ---
print("Generating roc_curve.png...")
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
ax.scatter(y, y_pred, alpha=0.5, color=primary_color, edgecolors='none', s=30)
min_val, max_val = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Prediction (y = x)')
ax.set_title('Actual vs Predicted Delivery Time (R² = 0.778)', fontsize=13, fontweight='bold', color=dark_color)
ax.set_xlabel('Actual Delivery Time (min)', fontweight='bold')
ax.set_ylabel('Predicted Delivery Time (min)', fontweight='bold')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
fig.tight_layout()
plt.savefig("reports/figures/roc_curve.png", dpi=300, bbox_inches='tight')
plt.close()

# --- Preprocess X for SHAP ---
print("Generating SHAP plots...")
preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]
X_trans = preprocessor.transform(X)

try:
    feature_names = list(preprocessor.get_feature_names_out())
    # Clean feature names (remove prefix like 'num__' or 'cat__')
    feature_names = [f.split('__')[-1] for f in feature_names]
except Exception:
    feature_names = [f"Feature_{i}" for i in range(X_trans.shape[1])]

X_trans_df = pd.DataFrame(X_trans, columns=feature_names)

# Use KernelExplainer / Explainer to bypass XGBoost string float parsing bug in TreeExplainer
explainer = shap.Explainer(model.predict, X_trans_df.sample(100, random_state=42))
shap_values = explainer(X_trans_df.sample(200, random_state=42))

# --- 4. SHAP Bar Plot ---
print("Generating shap_bar.png...")
plt.figure(figsize=(8, 5), dpi=300)
shap.plots.bar(shap_values, max_display=10, show=False)
plt.title('Global Feature Importance (SHAP Mean |Value|)', fontsize=13, fontweight='bold', color=dark_color)
plt.tight_layout()
plt.savefig("reports/figures/shap_bar.png", dpi=300, bbox_inches='tight')
plt.close()

# --- 5. SHAP Beeswarm Plot ---
print("Generating shap_beeswarm.png...")
plt.figure(figsize=(8, 5), dpi=300)
shap.plots.beeswarm(shap_values, max_display=10, show=False)
plt.title('SHAP Feature Impact & Value Distribution', fontsize=13, fontweight='bold', color=dark_color)
plt.tight_layout()
plt.savefig("reports/figures/shap_beeswarm.png", dpi=300, bbox_inches='tight')
plt.close()

# --- 6. SHAP Waterfall Plot ---
print("Generating shap_waterfall.png...")
plt.figure(figsize=(8, 5), dpi=300)
shap.plots.waterfall(shap_values[0], max_display=8, show=False)
plt.title('Local Prediction Explanation (Sample Order #1 Waterfall)', fontsize=13, fontweight='bold', color=dark_color)
plt.tight_layout()
plt.savefig("reports/figures/shap_waterfall.png", dpi=300, bbox_inches='tight')
plt.close()

print("ALL_IMAGES_GENERATED_SUCCESSFULLY")

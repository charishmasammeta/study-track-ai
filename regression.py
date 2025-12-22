import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ---------------- CONFIG ----------------
INPUT_FILE = "student_dataset.xlsx"
OUTPUT_EXCEL = "regression_scatter_output.xlsx"
OUTPUT_PNG = "regression_scatter_plot.png"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------------- 1. Load Dataset ----------------
df = pd.read_excel(INPUT_FILE)

# Map your dataset columns
study = "Study Hours"
work  = "Other Hours"
play  = "Play Hours"
sleep = "Sleep Hours"
marks = "Test Score"

# ---------------- 2. Prepare Data ----------------
X = df[[study, work, play, sleep]]
y = df[marks]

# ---------------- 3. Train Model ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

model = LinearRegression()
model.fit(X_train, y_train)

# ---------------- 4. Predict Marks ----------------
df["Predicted Marks"] = np.clip(model.predict(X).round(1), 0, 100)

# Evaluation
y_test_pred = model.predict(X_test)
r2 = r2_score(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
rmse = mse ** 0.5  # manual RMSE (works on all sklearn versions)

print("\n===== REGRESSION SUMMARY =====")
print("R² Score :", round(r2,4))
print("RMSE     :", round(rmse,4))
print("Coefficients:")
for feature, coef in zip([study, work, play, sleep], model.coef_):
    print(f"  {feature}: {round(coef,3)}")
print("Intercept:", round(model.intercept_,3))
print("================================\n")

# ---------------- 5. Scatter Plot ----------------
plt.figure(figsize=(7,7))
plt.scatter(df[marks], df["Predicted Marks"], color="blue", alpha=0.7)   # <--- HERE IS FIX

# Perfect reference line (y = x)
min_v = min(df[marks].min(), df["Predicted Marks"].min())
max_v = max(df[marks].max(), df["Predicted Marks"].max())
plt.plot([min_v, max_v], [min_v, max_v], color="red", linestyle="--", label="Perfect Fit Line")

plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Marks (Scatter Plot)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150)
plt.close()

print("Scatter plot saved as:", OUTPUT_PNG)

# ---------------- 6. Save Excel ----------------
df.to_excel(OUTPUT_EXCEL, index=False)
print("Excel saved as:", OUTPUT_EXCEL)

# Auto-open files (Windows only)
try:
    os.startfile(OUTPUT_EXCEL)
    os.startfile(OUTPUT_PNG)
except:
    pass

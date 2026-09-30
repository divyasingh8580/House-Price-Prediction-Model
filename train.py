import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 1. Load dataset
# -----------------------------
print("Loading California Housing dataset...")
housing = fetch_california_housing(as_frame=True)

df = housing.frame.copy()
df.rename(columns={"MedHouseVal": "target"}, inplace=True)

print("\nFirst 5 rows:")
print(df.head())

print("\nShape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# -----------------------------
# 2. Basic EDA
# -----------------------------
print("\nStatistical summary:")
print(df.describe())

os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.histplot(df["target"], kde=True)
plt.title("Distribution of House Prices")
plt.xlabel("Median House Value ($100,000)")
plt.tight_layout()
plt.savefig("plots/target_distribution.png")
plt.close()

# -----------------------------
# 3. Separate X and y
# -----------------------------
X = df.drop("target", axis=1)
y = df["target"]

# -----------------------------
# 4. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# -----------------------------
# 5. Preprocessing pipeline
# -----------------------------
preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# -----------------------------
# 6. Models
# -----------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=20),
    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )
}

results = []

best_model = None
best_rmse = float("inf")
best_name = None

# -----------------------------
# 7. Train and evaluate
# -----------------------------
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"\n{name}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_model = pipeline
        best_name = name

# -----------------------------
# 8. Model comparison
# -----------------------------
results_df = pd.DataFrame(results)
print("\nModel comparison:")
print(results_df)

results_df.to_csv("model_comparison.csv", index=False)

# -----------------------------
# 9. Save best model
# -----------------------------
os.makedirs("model", exist_ok=True)

joblib.dump(best_model, "model/house_price_model.pkl")
joblib.dump(list(X.columns), "model/feature_names.pkl")

print("\nBest model:", best_name)
print("Best RMSE:", best_rmse)
print("\nSaved:")
print("model/house_price_model.pkl")
print("model/feature_names.pkl")

# -----------------------------
# 10. Actual vs predicted plot
# -----------------------------
predictions = best_model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.4)
plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title(f"Actual vs Predicted — {best_name}")
plt.tight_layout()
plt.savefig("plots/actual_vs_predicted.png")
plt.close()

print("\nTraining completed successfully.")

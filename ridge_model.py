import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV


# Load data
train = pd.read_csv("cleaned_train.csv")
test = pd.read_csv("test.csv")

# Drop rows with missing values
train.dropna(inplace=True)
print(train.head())

# Separate features (X) and target (y)
# 'forward_returns' is the value we're trying to predict
X = train.drop(columns=["forward_returns"])
y = train["forward_returns"]

print("Target mean:", y.mean())
print("Target std:", y.std())

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (important for Ridge Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Initialize and train Ridge model
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = ridge.predict(X_val_scaled)
mse = mean_squared_error(y_val, y_pred)
print("Validation MSE:", mse)

# Experiment with different alpha values
for a in [0.01, 0.1, 1, 10, 100]:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train_scaled, y_train)
    preds = ridge.predict(X_val_scaled)
    mse = mean_squared_error(y_val, preds)
    print(f"alpha={a}, MSE={mse:.10f}")

# Compare first few predictions with actual values
# comparison = pd.DataFrame({"Actual": y_val.values[:10], "Predicted": y_pred[:10]})
# print(comparison)


# Perform Ridge regression with cross-validation
ridge_cv = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100], cv=5, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train_scaled, y_train)

print("\n--- Cross-Validation Results ---")
print("Best alpha from CV:", ridge_cv.alpha_)

# Evaluate on validation set
y_val_pred_cv = ridge_cv.predict(X_val_scaled)
mse_cv = mean_squared_error(y_val, y_val_pred_cv)

print("Validation MSE using best alpha:", mse_cv)

# Calculate R^2 scores
r2_simple = ridge.score(X_val_scaled, y_val)
print("Baseline Ridge R^2 (alpha=1.0):", r2_simple)

r2_cv = ridge_cv.score(X_val_scaled, y_val)
print("Validation R^2 using best alpha:", r2_cv)

# Show top 10 coefficients
coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": ridge_cv.coef_}).sort_values(by="Coefficient", ascending=False)
print("\nTop 10 features by coefficient:")
print(coef_df.head(10))



# Align columns between train and test
common_cols = X.columns.intersection(test.columns)
X_train = X_train[common_cols]
X_val = X_val[common_cols]
test = test[common_cols]
# Refit the scaler on the aligned training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Retrain RidgeCV on the aligned features
ridge_cv = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100], cv=5, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train_scaled, y_train)

# Scale and predict on the test set
X_test_scaled = scaler.transform(test)
test_preds = ridge_cv.predict(X_test_scaled)

# Save predictions to CSV
submission = pd.DataFrame({"Id": test.index, "Predicted": test_preds})
submission.to_csv("ridge_predictions.csv", index=False)


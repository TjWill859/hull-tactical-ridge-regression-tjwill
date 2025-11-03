# Hull Tactical Ridge Regression (DataSC Project)

This repository contains my individual contribution to the **DataSC (USC Data Science Club)** project for the [Hull Tactical Market Prediction Kaggle competition](https://www.kaggle.com/competitions/hull-tactical-market-prediction).

## 📈 Overview
The goal of the project was to predict **forward stock market returns** using historical financial and macroeconomic data.

My role focused on implementing and analyzing **Ridge Regression** using Python and scikit-learn.

## 🧩 Methods
- Used `Ridge` and `RidgeCV` to model and tune regularization.
- Standardized features using `StandardScaler`.
- Evaluated model performance with **Mean Squared Error (MSE)** and **R²**.
- Interpreted key feature coefficients to identify market indicators.

## 🧮 Example Results
- Best alpha from CV: 0.1
- Validation MSE using best alpha: 2.1639868079370684e-09
- Validation R^2 using best alpha: 0.9999813905113445

## 📂 Files
- `ridge_model.py` — Full implementation of the Ridge Regression model.
- `ridge_predictions.csv` — Example predictions on the test set.

## ⚖️ Note
This repo contains only my personal implementation, not the full team codebase, to respect DataSC project ownership.

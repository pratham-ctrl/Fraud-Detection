Author - Pratham Parikh

Credit Card Fraud Detection

Project: Credit Card Fraud Detection & Expense Analysis

Short description This project detects fraudulent credit-card transactions using a machine-learning pipeline (Random Forest by default) and provides an expense analysis tool that converts bank statement PDFs to CSV, analyzes spending, and visualizes expenses (Streamlit UI included).

Features

Data ingestion from CSV (Kaggle-style transaction dataset) and bank-statement PDFs (PDF → CSV converter).

Data preprocessing: cleaning, scaling, categorical encoding (if needed), and feature engineering.

Class imbalance handling: class weights and optional SMOTE oversampling.

Model training using Random Forest (configurable) and evaluation with precision, recall, F1-score, ROC-AUC, confusion matrix.

Model saving/loading (joblib/pickle) for inference.

Simple Streamlit dashboard for expense analysis and visualizations (monthly spend, category breakdown, top merchants).


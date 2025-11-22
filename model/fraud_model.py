import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from utils import evaluate_model # Assuming this is defined elsewhere
import streamlit as st
from sklearn.model_selection import train_test_split

class CreditCardFraudModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_path = "model/fraud_model.joblib"
        self.scaler_path = "model/scaler.joblib"

        # Load or create model
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.load_model()
        else:
            self.initialize_model()

    def initialize_model(self):
        """Initialize the model with default parameters"""
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42
        )
        self.scaler = StandardScaler()

    def train_on_real_data(self, df, test_size=0.2):
        """
        Train the model using real credit card transaction data

        Args:
            df (pandas.DataFrame): DataFrame containing transaction data
            test_size (float): Proportion of data to use for testing

        Returns:
            dict: Model performance metrics
        """
        try:
            # Extract features and target
            X = df.drop(['Class'], axis=1) if 'Class' in df.columns else df
            y = df['Class'] if 'Class' in df.columns else None

            if y is None:
                raise ValueError("No 'Class' column found in the dataset")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train model with progress bar
            with st.spinner("Training model... This may take a few minutes."):
                self.model.fit(X_train_scaled, y_train)

            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            y_prob = self.model.predict_proba(X_test_scaled)[:, 1]

            metrics = evaluate_model(y_test, y_pred, y_prob)

            # Save model
            self.save_model()

            return metrics

        except Exception as e:
            raise RuntimeError(f"Error training model: {str(e)}")

    def predict(self, features):
        """
        Predict fraud probability for new transactions

        Args:
            features (numpy.ndarray): Array of transaction features

        Returns:
            numpy.ndarray: Array of fraud probabilities
        """
        if self.model is None or self.scaler is None:
            self.initialize_model()

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Get probability predictions
        return self.model.predict_proba(features_scaled)[:, 1]

    def save_model(self):
        """Save the trained model and scaler"""
        os.makedirs("model", exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

    def load_model(self):
        """Load the trained model and scaler"""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
        except Exception as e:
            st.warning("Could not load saved model. Initializing new model.")
            self.initialize_model()

    def get_feature_importance(self):
        """
        Get the importance of each feature in the model

        Returns:
            dict: Feature names and their importance scores
        """
        if self.model is None:
            self.initialize_model()

        feature_names = ['Amount', 'Time', 'Frequency']
        importance_scores = self.model.feature_importances_

        return dict(zip(feature_names, importance_scores))

    def evaluate_transaction(self, amount, time, frequency):
        """
        Evaluate a single transaction

        Args:
            amount (float): Transaction amount
            time (float): Transaction time (in seconds from start of day)
            frequency (int): Number of transactions in recent history

        Returns:
            float: Fraud probability
        """
        features = np.array([[amount, time, frequency]])
        return self.predict(features)[0]
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class FraudDetector:
    def __init__(self):
        """Initialize the fraud detector with a pre-trained model"""
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        
        # For demonstration, we'll train a simple model
        # In production, you would load a pre-trained model
        self._train_demo_model()

    def _train_demo_model(self):
        """Train a simple model for demonstration"""
        # Generate synthetic data for demonstration
        np.random.seed(42)
        n_samples = 1000
        
        # Generate normal transactions
        normal_transactions = np.random.normal(100, 50, (n_samples, 2))
        
        # Generate fraudulent transactions
        fraud_transactions = np.random.normal(200, 100, (int(n_samples * 0.1), 2))
        
        # Combine the data
        X = np.vstack([normal_transactions, fraud_transactions])
        y = np.hstack([np.zeros(n_samples), np.ones(int(n_samples * 0.1))])
        
        # Fit the scaler and model
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, df):
        """Predict fraudulent transactions"""
        # Select relevant features (Amount and Time)
        features = df[['Amount', 'Time']].values
        
        # Scale the features
        features_scaled = self.scaler.transform(features)
        
        # Make predictions
        predictions = self.model.predict(features_scaled)
        
        return predictions

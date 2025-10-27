import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json
from datetime import datetime
import plotly.io as pio

def load_and_validate_csv(file):
    """Load and validate CSV file"""
    try:
        df = pd.read_csv(file)

        # Check required columns
        required_columns = {'Time', 'Amount'}
        if not required_columns.issubset(df.columns):
            raise ValueError("CSV must contain 'Time' and 'Amount' columns")

        # Basic data cleaning
        df = df.dropna(subset=['Time', 'Amount'])

        # Convert types
        df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

        return df

    except Exception as e:
        raise ValueError(f"Error processing CSV file: {str(e)}")

def load_training_data(file_paths):
    """Load and combine multiple CSV files for training"""
    dfs = []
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
        except Exception as e:
            st.error(f"Error loading {file_path}: {str(e)}")
            continue

    if not dfs:
        return None

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Basic preprocessing
    combined_df = combined_df.dropna()

    return combined_df

def evaluate_model(y_true, y_pred, y_prob=None):
    """Evaluate model performance"""
    report = classification_report(y_true, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)

    metrics = {
        'accuracy': report['accuracy'],
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1': report['1']['f1-score'],
        'confusion_matrix': conf_matrix
    }

    return metrics

def generate_analysis_report(transaction_data, fraud_predictions, username, format='json'):
    """
    Generate a comprehensive analysis report
    
    Args:
        transaction_data (pandas.DataFrame): Transaction data
        fraud_predictions (numpy.ndarray): Fraud prediction probabilities
        username (str): Username
        format (str): Output format ('json' or 'csv')
        
    Returns:
        tuple: (report_data, filename)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Basic statistics
    total_transactions = len(transaction_data)
    fraud_count = sum(fraud_predictions > 0.5)
    fraud_percentage = (fraud_count / total_transactions) * 100

    if format == 'csv':
        # Create a DataFrame for export
        export_df = transaction_data.copy()
        export_df['Fraud_Probability'] = fraud_predictions
        export_df['Is_Fraud'] = fraud_predictions > 0.5
        
        # Add metadata as header rows
        metadata = pd.DataFrame({
            'Info': ['Username', 'Generated At', 'Total Transactions', 'Fraud Count', 'Fraud Percentage'],
            'Value': [
                username,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                total_transactions,
                int(fraud_count),
                f"{round(fraud_percentage, 2)}%"
            ]
        })
        
        # Convert to CSV string
        metadata_csv = metadata.to_csv(index=False)
        transactions_csv = export_df.to_csv(index=False)
        
        # Combine with a separator
        csv_content = metadata_csv + "\n\n" + transactions_csv
        
        return csv_content, f"fraud_analysis_{timestamp}.csv"
    else:
        # JSON format (default)
        # Prepare report data
        report_data = {
            'report_info': {
                'username': username,
                'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_transactions': total_transactions,
                'fraud_count': int(fraud_count),
                'fraud_percentage': round(fraud_percentage, 2)
            },
            'transactions': []
        }

        # Add transaction details
        for idx, (_, row) in enumerate(transaction_data.iterrows()):
            report_data['transactions'].append({
                'time': float(row['Time']),
                'amount': float(row['Amount']),
                'fraud_probability': float(fraud_predictions[idx]),
                'is_fraud': bool(fraud_predictions[idx] > 0.5)
            })

        return report_data, f"fraud_analysis_{timestamp}.json"

def generate_model_performance_report(metrics, feature_importance, format='json'):
    """
    Generate a report of model performance metrics
    
    Args:
        metrics (dict): Model performance metrics
        feature_importance (dict): Feature importance scores
        format (str): Output format ('json' or 'csv')
        
    Returns:
        tuple: (report_data, filename)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if format == 'csv':
        # Create a DataFrame for model metrics
        metrics_df = pd.DataFrame({
            'Metric': ['Generated At', 'Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                f"{round(metrics['accuracy'] * 100, 2)}%",
                f"{round(metrics['precision'] * 100, 2)}%",
                f"{round(metrics['recall'] * 100, 2)}%",
                f"{round(metrics['f1'] * 100, 2)}%"
            ]
        })
        
        # Create a DataFrame for feature importance
        importance_df = pd.DataFrame({
            'Feature': list(feature_importance.keys()),
            'Importance': list(feature_importance.values())
        }).sort_values(by='Importance', ascending=False)
        
        # Convert confusion matrix to DataFrame
        cm_df = pd.DataFrame(
            metrics['confusion_matrix'],
            columns=['Predicted Normal', 'Predicted Fraud'],
            index=['Actual Normal', 'Actual Fraud']
        )
        
        # Convert to CSV format
        metrics_csv = metrics_df.to_csv(index=False)
        importance_csv = importance_df.to_csv(index=False)
        cm_csv = cm_df.to_csv()
        
        # Combine with section headers
        csv_content = "# MODEL PERFORMANCE METRICS\n\n" + metrics_csv + "\n\n"
        csv_content += "# FEATURE IMPORTANCE\n\n" + importance_csv + "\n\n"
        csv_content += "# CONFUSION MATRIX\n\n" + cm_csv
        
        return csv_content, f"model_performance_{timestamp}.csv"
    else:
        # JSON format (default)
        report_data = {
            'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_metrics': {
                'accuracy': round(metrics['accuracy'] * 100, 2),
                'precision': round(metrics['precision'] * 100, 2),
                'recall': round(metrics['recall'] * 100, 2),
                'f1_score': round(metrics['f1'] * 100, 2)
            },
            'feature_importance': {
                str(feature): float(importance)
                for feature, importance in feature_importance.items()
            },
            'confusion_matrix': metrics['confusion_matrix'].tolist()
        }

        return report_data, f"model_performance_{timestamp}.json"

def show_success(message):
    """Show success message"""
    st.success(message)

def show_error(message):
    """Show error message"""
    st.error(message)

def show_warning(message):
    """Show warning message"""
    st.warning(message)

def get_user_id(username):
    """Get user ID from username"""
    return username  # For simplicity, using username as ID

def save_transactions(df, user_id, fraud_probs):
    """Save transactions to database"""
    from models import Transaction, SessionLocal
    db = SessionLocal()
    try:
        for i, row in df.iterrows():
            transaction = Transaction(
                user_id=user_id,
                time=float(row['Time'].timestamp()),
                amount=float(row['Amount']),
                is_fraud=bool(fraud_probs[i] > 0.5)
            )
            db.add(transaction)
        db.commit()
    finally:
        db.close()

def get_user_transactions(user_id):
    """Get transactions for a user"""
    from models import Transaction, SessionLocal
    db = SessionLocal()
    try:
        return db.query(Transaction).filter(Transaction.user_id == user_id).all()
    finally:
        db.close()
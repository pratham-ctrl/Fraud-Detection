from auth import check_password, create_user, get_user_id, is_valid_email, is_valid_password

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import json
from utils import save_transactions, get_user_transactions, load_training_data, generate_analysis_report, generate_model_performance_report, show_success, show_error, show_warning
from indian_bank_parser import indian_bank_statement_to_df, display_indian_bank_transactions
from expense_analyzer import display_expense_analysis


def main():
    st.set_page_config(
        page_title="Credit Fraud Shield", 
        page_icon="💳", 
        layout="wide",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': "Credit Fraud Shield 🛡️"
        }
    )
    # Hide streamlit elements
    hide_streamlit_style = """
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {display: none !important;}
            div[data-testid="stToolbar"] {display: none !important;}
            .css-ch5dnh {display: none !important;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # Authentication
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if "username" not in st.session_state:
        st.session_state.username = ""

    if "model" not in st.session_state:
        from model.fraud_model import CreditCardFraudModel
        st.session_state.model = CreditCardFraudModel()

    # Login/Register page
    if not st.session_state.authenticated:
        # Custom CSS for full-screen login page
        login_page_style = """
        <style>
        div[data-testid="stVerticalBlock"] {
            padding-top: 2rem;
        }
        div[data-testid="stHorizontalBlock"] {
            padding: 2rem 0;
        }
        div.row-widget.stButton {
            text-align: center;
        }
        .login-container {
            max-width: 500px;
            margin: 0 auto;
            padding: 20px;
            border-radius: 10px;
            background-color: #f8f9fa;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        </style>
        """
        st.markdown(login_page_style, unsafe_allow_html=True)
        
        # Center the title
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.title("Credit Card Fraud Detection System")
            st.markdown("<h3 style='text-align: center;'>🛡️ Credit Fraud Shield 🛡️</h3>", unsafe_allow_html=True)
        
        # Create tabs for login and register
        login_tab, register_tab = st.tabs(["Login", "Register"])
        
        with login_tab:
            # Center the login form
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("<div class='login-container'>", unsafe_allow_html=True)
                st.subheader("Login to Your Account")
                
                email = st.text_input("Email Address", key="login_email")
                password = st.text_input("Password", type="password", key="login_password")
                
                login_clicked = st.button("Login", key="login_btn", use_container_width=True)
                
                if login_clicked:
                    if not email or not password:
                        st.error("Please enter both email and password.")
                    elif check_password(email, password):
                        st.session_state.authenticated = True
                        st.session_state.username = email
                        st.success(f"Welcome, {email}!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials. Please try again.")
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        with register_tab:
            # Center the registration form
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("<div class='login-container'>", unsafe_allow_html=True)
                st.subheader("Create a New Account")
                
                new_email = st.text_input("Email Address", key="register_email")
                new_password = st.text_input("Password", type="password", key="register_password")
                confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
                
                # Show password requirements
                if new_password:
                    if len(new_password) < 8:
                        st.warning("Password must be at least 8 characters long.")
                    else:
                        st.success("Password meets the minimum length requirement.")
                
                # Show email format requirements
                if new_email:
                    if not is_valid_email(new_email):
                        st.warning("Please enter a valid email address (must contain @).")
                    else:
                        st.success("Email format is valid.")
                
                register_clicked = st.button("Register", key="register_btn", use_container_width=True)
                
                if register_clicked:
                    if not new_email or not new_password or not confirm_password:
                        st.error("Please fill in all fields.")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match.")
                    elif not is_valid_email(new_email):
                        st.error("Please enter a valid email address.")
                    elif not is_valid_password(new_password):
                        st.error("Password must be at least 8 characters long.")
                    elif create_user(new_email, new_password):
                        st.success("✅ Registration successful!")
                        st.markdown("""
                        Your account has been created successfully. Please go to the login tab to sign in.
                        """)
                
                st.markdown("</div>", unsafe_allow_html=True)


    # Main app
    else:
        # Create columns for navbar
        col1, col2, col3 = st.columns([6,3,1])
        
        with col1:
            st.title("Credit Fraud Shield 🛡️")
        with col2:
            st.markdown(f"📧 {st.session_state.username}")
        with col3:
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.username = ""
                st.rerun()

        tabs = st.tabs(["Real-time Analysis", "Transaction History", "Model Insights"])

        with tabs[0]:
            st.markdown("### 💸 Real-time Transaction Analysis")
            
            # Add file upload options with tabs
            file_options = st.radio(
                "Choose data source:",
                ["Upload CSV", "Convert Bank Statement (PDF)"],
                horizontal=True
            )
            
            if file_options == "Upload CSV":
                uploaded_file = st.file_uploader("Upload transaction data (CSV)", type="csv")
                df = None
                
                if uploaded_file is not None:
                    try:
                        df = pd.read_csv(uploaded_file)
                    except Exception as e:
                        show_error(f"Error processing CSV file: {str(e)}")
            
            else:  # PDF conversion option
                st.markdown("### 📄 Convert Bank Statement PDF to CSV")
                st.markdown("""
                Upload your bank statement in PDF format to extract transaction data.
                The system will try to identify transaction details automatically.
                You can preview and adjust the data before analysis.
                """)
                
                pdf_file = st.file_uploader("Upload bank statement (PDF)", type="pdf")
                df = None
                
                if pdf_file is not None:
                    try:
                        with st.spinner("Extracting data from PDF..."):
                            df = indian_bank_statement_to_df(pdf_file)
                            
                            if df is not None and not df.empty:
                                st.success("✅ Successfully extracted transaction data from PDF!")
                                
                                # Display and allow for adjustments
                                df = display_indian_bank_transactions(df)
                                
                                st.warning("""
                                ⚠️ Please review the extracted data carefully. PDF extraction may not be 100% accurate 
                                and might require manual adjustments.
                                """)
                            else:
                                show_error("Could not extract transaction data from the PDF. Please try another file or upload a CSV instead.")
                                df = None
                    except Exception as e:
                        show_error(f"Error processing PDF file: {str(e)}")
                        df = None

            # Process uploaded data (either from CSV or PDF)
            if df is not None and not df.empty:
                try:
                    # Data preprocessing
                    if 'Time' in df.columns:
                        # Try to convert Time to datetime, handling different formats
                        try:
                            if df['Time'].dtype == 'object':  # String time format
                                df['Time'] = pd.to_datetime(df['Time'])
                            else:  # Numeric timestamp
                                df['Time'] = pd.to_datetime(df['Time'], unit='s')
                        except:
                            show_warning("Time format could not be processed. Using current time.")
                            df['Time'] = pd.to_datetime('now')
                    else:
                        show_warning("No Time column found. Using current time.")
                        df['Time'] = pd.to_datetime('now')
                    
                    # Ensure Amount is numeric
                    if 'Amount' in df.columns:
                        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
                    else:
                        show_error("No Amount column found. Cannot process transactions.")
                        return
                        
                    # Feature engineering
                    df['Hour'] = df['Time'].dt.hour
                    df['Weekday'] = df['Time'].dt.dayofweek

                    # Select relevant features for prediction
                    features = df[['Amount', 'Hour', 'Weekday']]

                    # Handle potential errors in the model
                    try:
                        # Get fraud probabilities
                        fraud_probs = st.session_state.model.predict(features)
                    except Exception as e:
                        show_error(f"Error during prediction: {str(e)}. Please check if your model is trained properly.")
                        return
                        
                    # Add fraud predictions to dataframe
                    df['Fraud Probability'] = fraud_probs
                    df['is_fraud'] = fraud_probs > 0.5

                    # Save to database
                    user_id = get_user_id(st.session_state.username)
                    save_transactions(df, user_id, fraud_probs)

                    # Display statistics in a modern card layout
                    st.markdown("### 📈 Transaction Overview")
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Total Transactions 📊", len(df))
                    with cols[1]:
                        fraud_count = len(df[df['is_fraud']])
                        st.metric("Fraudulent Transactions 🚨", fraud_count)
                    with cols[2]:
                        fraud_percentage = (fraud_count / len(df)) * 100
                        st.metric("Fraud Percentage ⚠️", f"{fraud_percentage:.2f}%")

                    # Enhanced visualizations
                    st.markdown("### 📊 Transaction Analysis")

                    # Amount Distribution with modern styling
                    fig1 = px.histogram(df, x="Amount", color="is_fraud",
                                         color_discrete_map={False: "#4B8BFF", True: "#FF4B4B"},
                                         labels={"is_fraud": "Fraud Status"},
                                         title="Distribution of Transaction Amounts")
                    fig1.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        title_font_size=20
                    )
                    st.plotly_chart(fig1, use_container_width=True)

                    # Fraud Probability Distribution
                    fig2 = px.histogram(df, x="Fraud Probability",
                                         title="Distribution of Fraud Probabilities",
                                         color_discrete_sequence=["#4B8BFF"])
                    fig2.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        title_font_size=20
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                    # Display fraudulent transactions in a modern table
                    if fraud_count > 0:
                        st.markdown("### 🚨 Fraudulent Transactions")
                        fraud_df = df[df['is_fraud']].sort_values(
                            by='Fraud Probability', ascending=False
                        )
                        st.dataframe(
                            fraud_df[['Time', 'Amount', 'Fraud Probability']],
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.success("✅ No fraudulent transactions detected!")
                    
                    # Add expense analysis if we have PDf data with Date, Debit, Credit columns
                    if file_options == "Convert Bank Statement (PDF)":
                        if all(col in df.columns for col in ['Date', 'Debit', 'Credit', 'Transaction Details']):
                            with st.expander("💰 Expense Management Suggestions", expanded=True):
                                display_expense_analysis(df)
                        else:
                            st.info("ℹ️ Expense analysis is only available for bank statements with proper date and transaction columns.")

                    # Add export functionality
                    st.markdown("### 📥 Export Analysis")
                    col1, col2 = st.columns(2)
                    
                    # JSON export
                    with col1:
                        if st.button("Export JSON Report", key="export_json"):
                            try:
                                # Generate JSON report
                                report_data, filename = generate_analysis_report(
                                    df, fraud_probs, st.session_state.username, format='json'
                                )

                                # Convert to JSON string
                                json_str = json.dumps(report_data, indent=2)

                                # Create download button
                                st.download_button(
                                    label="📥 Download JSON Report",
                                    data=json_str,
                                    file_name=filename,
                                    mime="application/json",
                                    key="download_json"
                                )
                                
                                show_success("JSON report generated successfully!")
                            except Exception as e:
                                show_error(f"Error generating JSON report: {str(e)}")
                    
                    # CSV export
                    with col2:
                        if st.button("Export CSV Report", key="export_csv"):
                            try:
                                # Generate CSV report
                                csv_data, filename = generate_analysis_report(
                                    df, fraud_probs, st.session_state.username, format='csv'
                                )

                                # Create download button
                                st.download_button(
                                    label="📥 Download CSV Report",
                                    data=csv_data,
                                    file_name=filename,
                                    mime="text/csv",
                                    key="download_csv"
                                )
                                
                                show_success("CSV report generated successfully!")
                            except Exception as e:
                                show_error(f"Error generating CSV report: {str(e)}")

                except Exception as e:
                    show_error(f"Error processing data: {str(e)}")

        with tabs[1]:
            st.markdown("### 📜 Transaction History")
            user_id = get_user_id(st.session_state.username)
            transactions = get_user_transactions(user_id)

            if transactions:
                df_history = pd.DataFrame([{
                    'Time': t.time,
                    'Amount': t.amount,
                    'Is Fraud': '🚨 Yes' if t.is_fraud else '✅ No',
                    'Date': t.created_at.strftime('%Y-%m-%d %H:%M:%S')
                } for t in transactions])

                st.dataframe(df_history, use_container_width=True, hide_index=True)
            else:
                st.info("📭 No transaction history found. Upload some transactions to get started!")

        with tabs[2]:
            st.markdown("### 🎯 Model Insights")

            # Add model training section
            st.markdown("#### 🔄 Train Model")
            st.markdown("""
            Upload credit card transaction datasets to train the model with real data.
            The CSV files should contain:
            - 'Time': Transaction time
            - 'Amount': Transaction amount
            - 'Class': Binary label (0 for normal, 1 for fraud)
            """)

            training_files = st.file_uploader(
                "Upload training data (CSV)",
                type="csv",
                accept_multiple_files=True,
                key="training_files"
            )

            if training_files:
                if st.button("Train Model", key="train_model"):
                    try:
                        with st.spinner("Loading and combining training data... 📊"):
                            combined_df = load_training_data(training_files)

                            if combined_df is not None:
                                st.info(f"Loaded {len(combined_df)} transactions for training")

                                # Train model and get metrics
                                metrics = st.session_state.model.train_on_real_data(combined_df)

                                # Display metrics in a modern layout
                                st.markdown("#### 📈 Training Results")
                                metric_cols = st.columns(4)

                                with metric_cols[0]:
                                    st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                                with metric_cols[1]:
                                    st.metric("Precision", f"{metrics['precision']:.2%}")
                                with metric_cols[2]:
                                    st.metric("Recall", f"{metrics['recall']:.2%}")
                                with metric_cols[3]:
                                    st.metric("F1 Score", f"{metrics['f1']:.2%}")

                                # Display confusion matrix
                                st.markdown("#### Confusion Matrix")
                                conf_matrix = metrics['confusion_matrix']
                                fig = px.imshow(
                                    conf_matrix,
                                    labels=dict(x="Predicted", y="Actual"),
                                    x=['Normal', 'Fraud'],
                                    y=['Normal', 'Fraud'],
                                    color_continuous_scale="RdBu",
                                    title="Confusion Matrix"
                                )
                                fig.update_layout(
                                    plot_bgcolor='white',
                                    paper_bgcolor='white'
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                # Add export functionality for model performance
                                st.markdown("### 📥 Export Model Performance Report")
                                exp_col1, exp_col2 = st.columns(2)
                                
                                # JSON export
                                with exp_col1:
                                    if st.button("Export JSON Report", key="perf_export_json"):
                                        try:
                                            # Generate JSON report
                                            report_data, filename = generate_model_performance_report(
                                                metrics,
                                                st.session_state.model.get_feature_importance(),
                                                format='json'
                                            )

                                            # Convert to JSON string
                                            json_str = json.dumps(report_data, indent=2)

                                            # Create download button
                                            st.download_button(
                                                label="📥 Download JSON Report",
                                                data=json_str,
                                                file_name=filename,
                                                mime="application/json",
                                                key="perf_download_json"
                                            )
                                            
                                            show_success("JSON report generated successfully!")
                                        except Exception as e:
                                            show_error(f"Error generating JSON report: {str(e)}")
                                
                                # CSV export
                                with exp_col2:
                                    if st.button("Export CSV Report", key="perf_export_csv"):
                                        try:
                                            # Generate CSV report
                                            csv_data, filename = generate_model_performance_report(
                                                metrics,
                                                st.session_state.model.get_feature_importance(),
                                                format='csv'
                                            )

                                            # Create download button
                                            st.download_button(
                                                label="📥 Download CSV Report",
                                                data=csv_data,
                                                file_name=filename,
                                                mime="text/csv",
                                                key="perf_download_csv"
                                            )
                                            
                                            show_success("CSV report generated successfully!")
                                        except Exception as e:
                                            show_error(f"Error generating CSV report: {str(e)}")

                                show_success("✅ Model successfully trained on real data!")
                            else:
                                show_error("❌ No valid training data found in the uploaded files")
                    except Exception as e:
                        show_error(f"Error during training: {str(e)}")

            # Feature importance with modern styling
            st.markdown("#### Feature Importance")
            importance = st.session_state.model.get_feature_importance()
            fig = px.bar(
                x=list(importance.keys()),
                y=list(importance.values()),
                title="Feature Importance in Fraud Detection",
                color_discrete_sequence=["#FF4B4B"]
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                title_font_size=20
            )
            st.plotly_chart(fig, use_container_width=True)

            # Model information with better formatting
            st.markdown("#### 🤖 Model Information")
            st.markdown("""
            This fraud detection model uses a Random Forest Classifier trained on transaction data.

            The model considers the following features:
            - 💰 **Transaction Amount**: Value of the transaction
            - ⏰ **Time of Transaction**: When the transaction occurred
            - 📊 **Transaction Frequency**: How often transactions occur

            The model assigns a probability score to each transaction, with higher scores indicating
            a greater likelihood of fraud. Transactions with a probability above 0.5 are flagged as fraudulent.
            """)

    if not st.session_state.authenticated:
        st.info("🔒 Please login or register to use the fraud detection system.")

if __name__ == "__main__":
    main()
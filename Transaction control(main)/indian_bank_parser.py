import os
import re
import io
import tempfile
from datetime import datetime
import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st

def extract_indian_bank_transactions(pdf_path):
    """
    Extract transactions specifically from Indian Bank statement PDF.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        pandas.DataFrame: DataFrame with transactions
    """
    transactions = []
    
    # Define specific patterns for Indian Bank format
    date_pattern = r'([A-Z][a-z]{2}\s+\d{1,2}\s+\d{4})'
    details_pattern = r'([A-Z0-9\s\/]+)'
    amount_pattern = r'(INR\s+[\d,]+\.\d{2})'
    balance_pattern = r'(INR\s+[\d,]+\.\d{2}\s+CR)'
    
    # Flag for transaction section
    in_transaction_section = False
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                
                lines = text.split('\n')
                
                for line_num, line in enumerate(lines):
                    # Check if we're in the transaction section (after the header row)
                    if "Date" in line and "Transaction Details" in line and "Debits" in line and "Credits" in line and "Balance" in line:
                        in_transaction_section = True
                        continue
                    
                    if "Ending Balance" in line:
                        in_transaction_section = False
                        continue
                    
                    if not in_transaction_section:
                        continue
                    
                    # Try to extract date - if found, it's a new transaction
                    date_match = re.search(date_pattern, line)
                    if date_match:
                        # Extract date
                        date_str = date_match.group(1)
                        try:
                            date = pd.to_datetime(date_str, format="%b %d %Y")
                        except:
                            continue
                        
                        # After the date, the next part is transaction details
                        details_start = date_match.end()
                        
                        # Look for "INR" patterns which indicate amounts
                        inr_positions = [m.start() for m in re.finditer(r'INR', line)]
                        
                        # Need at least two "INR" patterns (one for amount, one for balance)
                        if len(inr_positions) < 2:
                            continue
                        
                        # The transaction details is everything between date and first INR
                        details_end = inr_positions[0]
                        details = line[details_start:details_end].strip()
                        
                        # Identify columns based on their position
                        # In Indian Bank format:
                        # 1. First amount is either debit or credit (the other will be "-")
                        # 2. Last amount is always balance
                        
                        # Get all amount matches
                        amount_matches = re.finditer(amount_pattern, line)
                        amount_positions = []
                        
                        for match in amount_matches:
                            amount_positions.append({
                                'start': match.start(),
                                'end': match.end(),
                                'text': match.group(1)
                            })
                        
                        # Get the dash position if any
                        dash_pos = line.find("- ")
                        
                        # Initialize values
                        debit = 0.0
                        credit = 0.0
                        balance = 0.0
                        
                        # Last amount is balance
                        if amount_positions:
                            balance_text = amount_positions[-1]['text']
                            balance = parse_amount(balance_text)
                            
                            # The rest are either debit or credit
                            if len(amount_positions) > 1:
                                # Check which column has the dash
                                if dash_pos > 0:
                                    # Find which amount is closest to the dash
                                    debit_pos = None
                                    credit_pos = None
                                    
                                    for i, pos in enumerate(amount_positions[:-1]):  # Exclude balance
                                        if i == 0 and dash_pos > pos['end']:
                                            # If dash is after first amount, first is debit
                                            debit_pos = pos
                                        elif i == 0:
                                            # If dash is before first amount, second is credit
                                            credit_pos = pos
                                            
                                    if debit_pos:
                                        debit = parse_amount(debit_pos['text'])
                                    elif credit_pos:
                                        credit = parse_amount(credit_pos['text'])
                                else:
                                    # If no dash, first amount is typically debit in Indian Bank
                                    if len(amount_positions) > 1:
                                        debit = parse_amount(amount_positions[0]['text'])
                        
                        # Create transaction record
                        transactions.append({
                            'Date': date,
                            'Transaction Details': details,
                            'Debit': debit,
                            'Credit': credit,
                            'Balance': balance
                        })
    
    except Exception as e:
        st.error(f"Error extracting transactions: {str(e)}")
    
    # Convert to DataFrame
    if transactions:
        df = pd.DataFrame(transactions)
        # Add Time column
        df['Time'] = df['Date'].astype(np.int64) // 10**9
        # Calculate Amount
        df['Amount'] = df['Credit'] - df['Debit']
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        return df
    else:
        return pd.DataFrame(columns=['Date', 'Time', 'Transaction Details', 'Debit', 'Credit', 'Amount', 'Balance'])

def parse_amount(amount_str):
    """
    Parse amount specifically from Indian Bank format.
    
    Args:
        amount_str (str): Amount string (e.g., "INR 3,000.00")
        
    Returns:
        float: Parsed amount
    """
    # Remove "INR" and commas
    cleaned = amount_str.replace("INR", "").replace(",", "").strip()
    
    # Remove CR/DR indicators
    cleaned = cleaned.replace("CR", "").replace("DR", "").strip()
    
    try:
        return float(cleaned)
    except:
        return 0.0

def indian_bank_statement_to_df(pdf_file):
    """
    Convert an Indian Bank statement PDF to a transaction DataFrame.
    
    Args:
        pdf_file (UploadedFile): Streamlit uploaded file
        
    Returns:
        pandas.DataFrame: DataFrame with transactions
    """
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(pdf_file.getvalue())
        temp_path = temp_file.name
    
    try:
        # Process the PDF with approach tailored for Indian Bank
        st.info("Extracting Indian Bank statement data...")
        transactions_df = extract_indian_bank_transactions(temp_path)
        
        if not transactions_df.empty:
            st.success(f"Successfully extracted {len(transactions_df)} transactions from the Indian Bank statement.")
            
            # Clean up the data: ensure no row has both debit and credit
            for index, row in transactions_df.iterrows():
                if row['Debit'] > 0 and row['Credit'] > 0:
                    # Keep the larger value and set the other to zero
                    if row['Debit'] > row['Credit']:
                        transactions_df.at[index, 'Credit'] = 0.0
                    else:
                        transactions_df.at[index, 'Debit'] = 0.0
            
            return transactions_df
        else:
            st.error("No transactions could be extracted. Please check the PDF format.")
            return pd.DataFrame({
                'Date': [pd.to_datetime('today')],
                'Time': [pd.to_datetime('today').timestamp()],
                'Transaction Details': ["No transactions found"],
                'Debit': [0.0],
                'Credit': [0.0],
                'Amount': [0.0],
                'Balance': [0.0]
            })
    
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return pd.DataFrame({
            'Date': [pd.to_datetime('today')],
            'Time': [pd.to_datetime('today').timestamp()],
            'Transaction Details': ["Error processing PDF"],
            'Debit': [0.0],
            'Credit': [0.0],
            'Amount': [0.0],
            'Balance': [0.0]
        })
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def display_indian_bank_transactions(df):
    """
    Display transactions with edit options for Indian Bank format.
    
    Args:
        df (pandas.DataFrame): Transaction DataFrame
        
    Returns:
        pandas.DataFrame: Potentially modified DataFrame
    """
    st.subheader("Indian Bank Statement Transactions")
    
    # Ensure all the expected columns exist
    expected_columns = ['Date', 'Transaction Details', 'Debit', 'Credit', 'Amount', 'Balance']
    for col in expected_columns:
        if col not in df.columns:
            if col == 'Date':
                df[col] = pd.to_datetime('today')
            elif col in ['Debit', 'Credit', 'Amount', 'Balance']:
                df[col] = 0.0
            else:
                df[col] = ""
    
    # Format the display dataframe
    display_df = df.copy()
    if 'Date' in display_df.columns:
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    
    # Format monetary values with 2 decimal places and commas
    for col in ['Debit', 'Credit', 'Balance']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"₹ {x:,.2f}" if x > 0 else "")
    
    # Show data in bank statement format
    display_columns = ['Date', 'Transaction Details', 'Debit', 'Credit', 'Balance']
    st.dataframe(display_df[display_columns], use_container_width=True)
    
    # Tabs for editing and export
    tabs = st.tabs(["Add Transaction", "Edit/Delete", "Export"])
    
    with tabs[0]:
        st.subheader("Add New Transaction")
        
        with st.form("add_transaction"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_date = st.date_input("Date", value=pd.to_datetime('today'))
                transaction_type = st.radio("Type", ["Debit (Payment)", "Credit (Deposit)"])
            
            with col2:
                amount = st.number_input("Amount", min_value=0.0, step=0.01, format="%.2f")
                balance = st.number_input("Balance", step=0.01, format="%.2f")
            
            details = st.text_input("Transaction Details")
            
            submit = st.form_submit_button("Add Transaction")
            
            if submit:
                # Create new row
                new_row = {
                    'Date': pd.to_datetime(new_date),
                    'Time': pd.to_datetime(new_date).timestamp(),
                    'Transaction Details': details,
                    'Balance': balance
                }
                
                if transaction_type.startswith("Debit"):
                    new_row['Debit'] = amount
                    new_row['Credit'] = 0.0
                    new_row['Amount'] = -amount
                else:
                    new_row['Debit'] = 0.0
                    new_row['Credit'] = amount
                    new_row['Amount'] = amount
                
                # Add to DataFrame
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                df = df.sort_values('Date').reset_index(drop=True)
                
                st.success("Transaction added!")
                
                # Update display
                display_df = df.copy()
                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                for col in ['Debit', 'Credit', 'Balance']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"₹ {x:,.2f}" if x > 0 else "")
                st.dataframe(display_df[display_columns], use_container_width=True)
    
    with tabs[1]:
        st.subheader("Edit or Delete Transactions")
        
        if st.checkbox("Enable filtering"):
            # Date filter
            if 'Date' in df.columns:
                min_date = df['Date'].min().date()
                max_date = df['Date'].max().date()
                date_range = st.date_input("Date range", value=(min_date, max_date))
                
                if len(date_range) == 2:
                    df_filtered = df[
                        (df['Date'].dt.date >= date_range[0]) &
                        (df['Date'].dt.date <= date_range[1])
                    ]
                else:
                    df_filtered = df
            else:
                df_filtered = df
            
            # Transaction type filter
            transaction_types = []
            col1, col2 = st.columns(2)
            with col1:
                if st.checkbox("Include deposits", value=True):
                    transaction_types.append("Credit")
            with col2:
                if st.checkbox("Include payments", value=True):
                    transaction_types.append("Debit")
            
            if transaction_types:
                if "Credit" in transaction_types and "Debit" not in transaction_types:
                    df_filtered = df_filtered[df_filtered['Credit'] > 0]
                elif "Debit" in transaction_types and "Credit" not in transaction_types:
                    df_filtered = df_filtered[df_filtered['Debit'] > 0]
            
            # Display filtered data
            st.write(f"Showing {len(df_filtered)} of {len(df)} transactions")
            
            # Display with formatted values
            display_filtered = df_filtered.copy()
            display_filtered['Date'] = display_filtered['Date'].dt.strftime('%Y-%m-%d')
            for col in ['Debit', 'Credit', 'Balance']:
                if col in display_filtered.columns:
                    display_filtered[col] = display_filtered[col].apply(lambda x: f"₹ {x:,.2f}" if x > 0 else "")
            st.dataframe(display_filtered[display_columns], use_container_width=True)
            
            # Delete option
            if st.button("Delete filtered transactions"):
                df = df.drop(df_filtered.index).reset_index(drop=True)
                st.success(f"Deleted {len(df_filtered)} transactions")
                
                # Update display
                display_df = df.copy()
                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                for col in ['Debit', 'Credit', 'Balance']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"₹ {x:,.2f}" if x > 0 else "")
                st.dataframe(display_df[display_columns], use_container_width=True)
    
    with tabs[2]:
        st.subheader("Export Data")
        
        export_format = st.radio("Format", ["CSV", "Excel"])
        
        if export_format == "CSV":
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                data=csv,
                file_name="indian_bank_transactions.csv",
                mime="text/csv"
            )
        else:
            # Create Excel file in memory
            buffer = io.BytesIO()
            
            try:
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    # Convert datetime to string for Excel
                    excel_df = df.copy()
                    excel_df['Date'] = excel_df['Date'].dt.strftime('%Y-%m-%d')
                    
                    # Write to Excel
                    excel_df.to_excel(writer, sheet_name='Transactions', index=False)
                    
                    # Get workbook and worksheet objects
                    workbook = writer.book
                    worksheet = writer.sheets['Transactions']
                    
                    # Add formats
                    money_format = workbook.add_format({'num_format': '₹ #,##0.00'})
                    date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})
                    
                    # Apply formats to columns
                    worksheet.set_column('A:A', 12, date_format)  # Date
                    worksheet.set_column('C:F', 15, money_format)  # Money columns
                    worksheet.set_column('B:B', 40)  # Transaction details
                
                # Get Excel data
                buffer.seek(0)
                
                st.download_button(
                    "Download Excel",
                    data=buffer,
                    file_name="indian_bank_transactions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Error creating Excel file: {str(e)}")
                st.info("Please install xlsxwriter package to enable Excel export.")
    
    return df
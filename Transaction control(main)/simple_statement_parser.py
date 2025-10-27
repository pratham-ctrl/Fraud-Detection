import os
import re
import io
import tempfile
from datetime import datetime
import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st

def extract_transactions_simple(pdf_path):
    """
    Extract transactions from a bank statement PDF with improved pattern matching
    specifically for Indian Bank statements.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        pandas.DataFrame: DataFrame with transactions
    """
    transactions = []
    
    # Define improved patterns for Indian Bank format
    # Format: "Feb 10 2025" or similar
    date_pattern = r'([A-Z][a-z]{2}\s+\d{1,2}\s+\d{4})'
    
    # Format: "INR 3,000.00" or similar
    amount_pattern = r'(INR\s+[\d,]+\.\d{2})'
    
    # Balance format: "INR 5,903.23 CR" or similar
    balance_pattern = r'(INR\s+[\d,]+\.\d{2}\s+(?:CR|DR))'
    
    # Transaction in progress flag
    transaction_in_progress = False
    current_transaction = {}
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Process each page
            for page_num, page in enumerate(pdf.pages):
                # Extract text
                text = page.extract_text()
                if not text:
                    continue
                
                # Split into lines
                lines = text.split('\n')
                
                # Track potential transaction rows
                for line_num, line in enumerate(lines):
                    # Skip header rows and summary sections
                    if "Date" in line and "Transaction Details" in line and "Balance" in line:
                        continue
                    if "ACCOUNT DETAILS" in line or "ACCOUNT SUMMARY" in line:
                        continue
                    if "Opening Balance" in line or "Total Credits" in line:
                        continue
                    if "ACCOUNT ACTIVITY" in line or "Ending Balance" in line:
                        continue
                    if "Total Debits" in line or "Total" in line:
                        continue
                    
                    # If line contains a date, it might be the start of a transaction
                    date_match = re.search(date_pattern, line)
                    if date_match:
                        # If we were processing a transaction, save it before starting a new one
                        if transaction_in_progress and current_transaction:
                            transactions.append(current_transaction.copy())
                            
                        # Start a new transaction
                        transaction_in_progress = True
                        current_transaction = {
                            'Date': None,
                            'Transaction Details': '',
                            'Debit': 0.0,
                            'Credit': 0.0,
                            'Balance': 0.0
                        }
                        
                        # Parse date
                        date_str = date_match.group(1)
                        try:
                            date = pd.to_datetime(date_str)
                            current_transaction['Date'] = date
                        except:
                            # Try with explicit format for "Feb 10 2025"
                            try:
                                date = pd.to_datetime(date_str, format="%b %d %Y")
                                current_transaction['Date'] = date
                            except:
                                # Skip if we can't parse the date
                                transaction_in_progress = False
                                continue
                        
                        # Extract transaction details (everything after date until first amount)
                        details_start = date_match.span()[1]
                        
                        # Find positions of amounts in the line
                        debit_match = re.search(amount_pattern, line[details_start:])
                        credit_match = None
                        balance_match = re.search(balance_pattern, line)
                        
                        # Extract transaction details
                        if debit_match:
                            details_end = details_start + debit_match.span()[0]
                            current_transaction['Transaction Details'] = line[details_start:details_end].strip()
                            
                            # Extract debit amount
                            debit_str = debit_match.group(1)
                            current_transaction['Debit'] = parse_amount(debit_str)
                            
                            # Find credit amount (if any) after debit
                            remaining_line = line[details_start + debit_match.span()[1]:]
                            credit_match = re.search(amount_pattern, remaining_line)
                        else:
                            # If no debit, look for credit
                            credit_match = re.search(amount_pattern, line[details_start:])
                            if credit_match:
                                details_end = details_start + credit_match.span()[0]
                                current_transaction['Transaction Details'] = line[details_start:details_end].strip()
                        
                        # Extract credit amount if found
                        if credit_match:
                            credit_str = credit_match.group(1)
                            current_transaction['Credit'] = parse_amount(credit_str)
                        
                        # Extract balance amount if found
                        if balance_match:
                            balance_str = balance_match.group(1)
                            current_transaction['Balance'] = parse_amount(balance_str)
                            
                    # If no date but we're processing a transaction, this might be a continuation of transaction details
                    elif transaction_in_progress:
                        # Check if this line has amounts but no date
                        if re.search(amount_pattern, line) and not re.search(date_pattern, line):
                            # This might be a continuation with amounts
                            balance_match = re.search(balance_pattern, line)
                            
                            # Look for debit and credit
                            amounts = re.findall(amount_pattern, line)
                            
                            if amounts:
                                # First check if we have a full row with columns
                                if len(amounts) >= 3:
                                    # In the Indian Bank format, if a line has 3 amount fields:
                                    # 1. First is either debit or credit (one is empty with a "-" placeholder)
                                    # 2. Second is either debit or credit (the other one)
                                    # 3. Third is the balance
                                    
                                    # Check if the line has "- " which indicates no value in that column
                                    if "- " in line:
                                        # Determine which column has the dash
                                        if line.find("- ") < line.find(amounts[0]):
                                            # The dash is before the first amount, meaning first amount is credit
                                            current_transaction['Credit'] = parse_amount(amounts[0])
                                            current_transaction['Debit'] = 0.0
                                        else:
                                            # The dash is after first amount, meaning first amount is debit
                                            current_transaction['Debit'] = parse_amount(amounts[0])
                                            current_transaction['Credit'] = 0.0
                                    else:
                                        # If no dash, use the position to determine
                                        # In Indian Bank format, debit is typically first column
                                        current_transaction['Debit'] = parse_amount(amounts[0])
                                        current_transaction['Credit'] = 0.0
                                elif len(amounts) == 2:
                                    # Usually in format [debit, -] or [-, credit]
                                    if "- " in line:
                                        # Check which amount is near the dash
                                        dash_pos = line.find("- ")
                                        if abs(dash_pos - line.find(amounts[0])) < abs(dash_pos - line.find(amounts[1])):
                                            # First amount is closer to dash, so second is the value
                                            current_transaction['Credit'] = parse_amount(amounts[1])
                                            current_transaction['Debit'] = 0.0
                                        else:
                                            # Second amount is closer to dash, so first is the value
                                            current_transaction['Debit'] = parse_amount(amounts[0])
                                            current_transaction['Credit'] = 0.0
                                    else:
                                        # If no dash, assume first is debit (for Indian Bank)
                                        current_transaction['Debit'] = parse_amount(amounts[0])
                                        current_transaction['Credit'] = 0.0
                                elif len(amounts) == 1:
                                    # For single amount, check the line context
                                    line_lower = line.lower()
                                    if "credit" in line_lower or "deposit" in line_lower:
                                        current_transaction['Credit'] = parse_amount(amounts[0])
                                        current_transaction['Debit'] = 0.0
                                    else:
                                        # Default to debit for Indian Bank (as most entries are debits)
                                        current_transaction['Debit'] = parse_amount(amounts[0])
                                        current_transaction['Credit'] = 0.0
                            
                            # Extract balance if found
                            if balance_match:
                                current_transaction['Balance'] = parse_amount(balance_match.group(1))
                                
                                # Add transaction and reset
                                transactions.append(current_transaction.copy())
                                transaction_in_progress = False
                        else:
                            # This is probably a continuation of the transaction details
                            current_transaction['Transaction Details'] += " " + line.strip()
                
                # If we have a transaction in progress at the end of the page, add it
                if transaction_in_progress and current_transaction:
                    transactions.append(current_transaction.copy())
                    transaction_in_progress = False
    
    except Exception as e:
        st.error(f"Error in simple extraction: {str(e)}")
    
    # Convert to DataFrame
    if transactions:
        df = pd.DataFrame(transactions)
        # Add Time column
        df['Time'] = df['Date'].astype(np.int64) // 10**9
        # Calculate Amount as Credit - Debit
        df['Amount'] = df['Credit'] - df['Debit']
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        return df
    else:
        # Return empty DataFrame with required columns
        return pd.DataFrame(columns=['Date', 'Time', 'Transaction Details', 'Debit', 'Credit', 'Amount', 'Balance'])

def parse_amount(amount_str):
    """
    Parse amount string to float, with specific handling for Indian Bank format.
    
    Args:
        amount_str (str): Amount string (e.g., "INR 3,000.00" or "INR 5,903.23 CR")
        
    Returns:
        float: Parsed amount
    """
    if not amount_str:
        return 0.0
    
    # Handle Indian Bank format with "INR" and CR/DR indicators
    is_credit = "CR" in amount_str
    is_debit = "DR" in amount_str
    
    # Remove currency symbols, "INR", and commas
    cleaned = re.sub(r'[INR$,]', '', amount_str)
    
    # Remove CR/DR indicators
    cleaned = cleaned.replace("CR", "").replace("DR", "")
    
    # Check for other explicit negative indicators
    is_negative = '-' in cleaned
    
    # Remove all non-numeric characters except decimal point
    cleaned = re.sub(r'[^0-9.\-]', '', cleaned)
    
    try:
        amount = float(cleaned)
        
        # Apply negative for debit entries if amount is positive
        if (is_debit or is_negative) and amount > 0:
            amount = -amount
            
        return amount
    except:
        return 0.0

def convert_statement_to_transactions(pdf_file):
    """
    Convert a bank statement PDF to a DataFrame of transactions.
    
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
        # Process the PDF with direct approach
        st.info("Extracting bank statement data...")
        transactions_df = extract_transactions_simple(temp_path)
        
        if not transactions_df.empty:
            st.success(f"Successfully extracted {len(transactions_df)} transactions from the statement.")
            return transactions_df
        else:
            st.error("No transactions could be extracted. The PDF format may not be supported.")
            # Return minimal DataFrame as fallback
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
        # Return minimal DataFrame as fallback
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

def display_transactions(df):
    """
    Display transactions with edit options.
    
    Args:
        df (pandas.DataFrame): Transaction DataFrame
        
    Returns:
        pandas.DataFrame: Potentially modified DataFrame
    """
    st.subheader("Bank Statement Transactions")
    
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
    
    # Create a display version with formatted Date
    display_df = df.copy()
    if 'Date' in display_df.columns:
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    
    # Show data in bank statement format
    display_columns = ['Date', 'Transaction Details', 'Debit', 'Credit', 'Balance']
    st.dataframe(display_df[display_columns])
    
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
                st.dataframe(display_df[display_columns])
    
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
            
            # Display with formatted date
            display_filtered = df_filtered.copy()
            display_filtered['Date'] = display_filtered['Date'].dt.strftime('%Y-%m-%d')
            st.dataframe(display_filtered[display_columns])
            
            # Delete option
            if st.button("Delete filtered transactions"):
                df = df.drop(df_filtered.index).reset_index(drop=True)
                st.success(f"Deleted {len(df_filtered)} transactions")
                
                # Update display
                display_df = df.copy()
                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                st.dataframe(display_df[display_columns])
    
    with tabs[2]:
        st.subheader("Export Data")
        
        export_format = st.radio("Format", ["CSV", "Excel"])
        
        if export_format == "CSV":
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                data=csv,
                file_name="bank_transactions.csv",
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
                    money_format = workbook.add_format({'num_format': '#,##0.00'})
                    date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})
                    
                    # Apply formats to columns
                    worksheet.set_column('A:A', 12, date_format)  # Date
                    worksheet.set_column('C:F', 12, money_format)  # Money columns
                    worksheet.set_column('B:B', 40)  # Transaction details
                
                # Get Excel data
                buffer.seek(0)
                
                st.download_button(
                    "Download Excel",
                    data=buffer,
                    file_name="bank_transactions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Error creating Excel file: {str(e)}")
                st.info("Please install xlsxwriter package to enable Excel export.")
    
    return df
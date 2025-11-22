import os
import re
import io
import tempfile
from datetime import datetime
import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st

def extract_transactions_from_bank_statement(pdf_path):
    """
    Extract transactions from bank statement PDF with proper column separation.
    
    This function uses a combined approach specifically for bank statements:
    1. First tries to identify the table structure in the PDF
    2. Uses positional analysis to separate columns if tables aren't detected
    3. Applies bank-specific patterns to identify transaction rows
    
    Args:
        pdf_path (str): Path to the PDF file
    
    Returns:
        pandas.DataFrame: DataFrame with properly separated columns
    """
    all_transactions = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            st.info(f"Processing {page_count} pages from PDF statement")
            
            # Process each page
            for page_num, page in enumerate(pdf.pages):
                st.write(f"Processing page {page_num+1}/{page_count}")
                
                # Try to extract tables first
                tables = page.extract_tables()
                if tables and len(tables) > 0:
                    # Tables found - process them
                    for table in tables:
                        # Check if table has the expected structure
                        if len(table) > 1:  # At least one row plus header
                            transactions = process_bank_table(table)
                            if transactions:
                                all_transactions.extend(transactions)
                
                # If no transactions found with tables, try positional analysis
                if not all_transactions and page_num < 3:  # Only try on first few pages if tables failed
                    transactions = process_by_position(page)
                    if transactions:
                        all_transactions.extend(transactions)
        
        if all_transactions:
            # Convert to DataFrame
            df = pd.DataFrame(all_transactions)
            
            # Ensure we have standard columns with proper types
            df = standardize_transaction_columns(df)
            
            return df
        else:
            st.warning("No transactions found in the PDF. Please try a different statement.")
            return pd.DataFrame()
    
    except Exception as e:
        st.error(f"Error extracting transactions: {str(e)}")
        return pd.DataFrame()

def process_bank_table(table):
    """
    Process a table extracted from a bank statement.
    
    Args:
        table (list): List of rows where each row is a list of cell values
    
    Returns:
        list: List of transaction dictionaries
    """
    transactions = []
    
    # Try to identify header row to determine column positions
    header_row = table[0]
    column_indices = identify_columns(header_row)
    
    # If column detection failed, try generic approach
    if not any(column_indices.values()):
        # No clear headers, try to detect by content
        for row in table[1:]:  # Skip potential header
            # Skip empty or non-transaction rows
            if not row or len(row) < 3 or all(cell is None or cell == "" for cell in row):
                continue
                
            transaction = identify_columns_by_content(row)
            if transaction.get('Date') and (transaction.get('Debit') or transaction.get('Credit')):
                transactions.append(transaction)
    else:
        # Headers identified, use column positions
        for row in table[1:]:  # Skip header
            # Skip empty rows
            if not row or len(row) < 3 or all(cell is None or cell == "" for cell in row):
                continue
                
            transaction = {}
            
            for col_name, col_index in column_indices.items():
                if col_index is not None and col_index < len(row):
                    cell_value = row[col_index]
                    
                    # Convert based on column type
                    if col_name == 'Date' and cell_value:
                        transaction['Date'] = parse_date(cell_value)
                    elif col_name in ['Debit', 'Credit', 'Amount', 'Balance']:
                        transaction[col_name] = parse_amount(cell_value)
                    elif col_name == 'Transaction Details':
                        transaction[col_name] = str(cell_value) if cell_value else ""
            
            # Only add if we have key transaction info
            if transaction.get('Date') and (transaction.get('Debit') or transaction.get('Credit') or transaction.get('Amount')):
                transactions.append(transaction)
    
    return transactions

def identify_columns(header_row):
    """
    Identify column indices from header row.
    
    Args:
        header_row (list): List of header cell values
    
    Returns:
        dict: Dictionary of column names to indices
    """
    # Map of possible header names to standardized column names
    header_mapping = {
        'date': 'Date',
        'time': 'Date',
        'posted': 'Date',
        'transaction date': 'Date',
        'description': 'Transaction Details',
        'details': 'Transaction Details',
        'transaction': 'Transaction Details',
        'particulars': 'Transaction Details',
        'narration': 'Transaction Details',
        'debit': 'Debit',
        'withdrawal': 'Debit',
        'withdraw': 'Debit',
        'payments': 'Debit',
        'out': 'Debit',
        'money out': 'Debit',
        'credit': 'Credit',
        'deposit': 'Credit',
        'deposits': 'Credit',
        'in': 'Credit',
        'money in': 'Credit',
        'amount': 'Amount',
        'balance': 'Balance'
    }
    
    column_indices = {
        'Date': None,
        'Transaction Details': None,
        'Debit': None,
        'Credit': None,
        'Amount': None,
        'Balance': None
    }
    
    # Check for headers
    if header_row:
        for i, cell in enumerate(header_row):
            if not cell:
                continue
                
            cell_lower = str(cell).lower().strip()
            
            # Check if it matches any known header
            for key, value in header_mapping.items():
                if key in cell_lower:
                    column_indices[value] = i
                    break
    
    return column_indices

def identify_columns_by_content(row):
    """
    Identify column content without relying on headers.
    
    Args:
        row (list): List of cell values
    
    Returns:
        dict: Transaction dictionary with identified fields
    """
    transaction = {
        'Date': None,
        'Transaction Details': "",
        'Debit': 0.0,
        'Credit': 0.0,
        'Amount': None,
        'Balance': None
    }
    
    # Date is usually in first 2 columns
    for i in range(min(2, len(row))):
        if row[i] and isinstance(row[i], str):
            date = parse_date(row[i])
            if date:
                transaction['Date'] = date
                break
    
    # Amounts are usually numeric and in rightmost columns
    amount_cols = []
    for i in range(len(row)):
        if row[i] and isinstance(row[i], (str, int, float)):
            amount = parse_amount(row[i])
            if amount is not None:
                amount_cols.append((i, amount))
    
    # Sort amounts by position (right to left)
    amount_cols.sort(key=lambda x: x[0], reverse=True)
    
    # Assign amounts based on position
    if len(amount_cols) >= 3:  # Likely has Debit, Credit, Balance
        # Rightmost is usually balance
        transaction['Balance'] = amount_cols[0][1]
        
        # Check the signs of the next two
        if amount_cols[1][1] < 0:
            transaction['Debit'] = abs(amount_cols[1][1])
        else:
            transaction['Credit'] = amount_cols[1][1]
            
        if amount_cols[2][1] < 0:
            transaction['Debit'] = abs(amount_cols[2][1])
        else:
            transaction['Credit'] = amount_cols[2][1]
            
    elif len(amount_cols) == 2:  # Likely has Amount and Balance
        transaction['Balance'] = amount_cols[0][1]
        amount = amount_cols[1][1]
        
        # Split into debit/credit based on sign
        if amount < 0:
            transaction['Debit'] = abs(amount)
        else:
            transaction['Credit'] = amount
        transaction['Amount'] = amount
        
    elif len(amount_cols) == 1:  # Just an amount
        amount = amount_cols[0][1]
        if amount < 0:
            transaction['Debit'] = abs(amount)
        else:
            transaction['Credit'] = amount
        transaction['Amount'] = amount
    
    # Transaction details are everything that's not a date or amount
    details = []
    for i, cell in enumerate(row):
        if cell and isinstance(cell, str) and not parse_date(cell) and parse_amount(cell) is None:
            details.append(str(cell).strip())
    
    transaction['Transaction Details'] = " ".join(details)
    
    return transaction

def process_by_position(page):
    """
    Process a page by analyzing text positions to determine columns.
    
    Args:
        page (pdfplumber.page.Page): PDF page object
    
    Returns:
        list: List of transaction dictionaries
    """
    transactions = []
    
    # Extract text with positions
    words = page.extract_words(x_tolerance=3, y_tolerance=3)
    
    if not words:
        return transactions
    
    # Group words by line
    lines = {}
    for word in words:
        # Round y position to group lines
        y_pos = round(word['top'])
        if y_pos not in lines:
            lines[y_pos] = []
        lines[y_pos].append(word)
    
    # Sort lines by y position
    sorted_lines = sorted(lines.items())
    
    # Analyze x-positions to determine likely columns
    x_positions = analyze_x_positions(sorted_lines)
    
    # If we have at least 3 columns, process them
    if len(x_positions) >= 3:
        # Process each line to extract transactions
        current_transaction = {}
        
        for y_pos, line_words in sorted_lines:
            # Sort words by x position
            line_words.sort(key=lambda w: w['x0'])
            
            # Skip lines that don't have enough words
            if len(line_words) < 2:
                continue
            
            # Check if this looks like a transaction line
            line_text = " ".join([w['text'] for w in line_words])
            
            # Look for date pattern
            date_match = re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', line_text)
            if date_match:
                # If we already have a transaction, save it
                if current_transaction.get('Date') and (
                    current_transaction.get('Debit', 0) > 0 or 
                    current_transaction.get('Credit', 0) > 0 or
                    current_transaction.get('Amount')
                ):
                    transactions.append(current_transaction)
                
                # Start a new transaction
                current_transaction = {
                    'Date': parse_date(date_match.group(0)),
                    'Transaction Details': "",
                    'Debit': 0.0,
                    'Credit': 0.0,
                    'Amount': None,
                    'Balance': None
                }
                
                # Extract other transaction data based on x positions
                extract_transaction_data_by_position(current_transaction, line_words, x_positions)
            
            # If this is part of an existing transaction, add to details
            elif current_transaction.get('Date'):
                # Check if line has amounts
                has_amounts = extract_transaction_data_by_position(current_transaction, line_words, x_positions)
                
                # If no amounts found, this might be continuation of description
                if not has_amounts:
                    if current_transaction['Transaction Details']:
                        current_transaction['Transaction Details'] += " " + line_text
                    else:
                        current_transaction['Transaction Details'] = line_text
        
        # Add the last transaction if we have one
        if current_transaction.get('Date') and (
            current_transaction.get('Debit', 0) > 0 or 
            current_transaction.get('Credit', 0) > 0 or
            current_transaction.get('Amount')
        ):
            transactions.append(current_transaction)
    
    return transactions

def analyze_x_positions(sorted_lines):
    """
    Analyze x positions to determine column structure.
    
    Args:
        sorted_lines (list): List of (y_pos, words) tuples
    
    Returns:
        list: List of column x positions
    """
    # Count occurrences of each x position to find columns
    x_counts = {}
    
    for _, line_words in sorted_lines:
        for word in line_words:
            x_pos = round(word['x0'] / 10) * 10  # Round to nearest 10 to group close positions
            if x_pos not in x_counts:
                x_counts[x_pos] = 0
            x_counts[x_pos] += 1
    
    # Get most common x positions (columns)
    common_positions = sorted(x_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Take the top 4-5 positions
    top_positions = [pos for pos, count in common_positions[:5] if count > len(sorted_lines) * 0.2]
    
    # Sort by position
    return sorted(top_positions)

def extract_transaction_data_by_position(transaction, line_words, x_positions):
    """
    Extract transaction data from a line based on position analysis.
    
    Args:
        transaction (dict): Transaction dictionary to update
        line_words (list): List of word dictionaries for the line
        x_positions (list): List of column x positions
    
    Returns:
        bool: True if amounts were found, False otherwise
    """
    has_amounts = False
    
    # Check each word to see which column it belongs to
    for word in line_words:
        word_text = word['text']
        word_x = word['x0']
        
        # Find which column this word belongs to
        col_index = None
        for i, pos in enumerate(x_positions):
            if abs(word_x - pos) < 50:  # Within 50 points of column position
                col_index = i
                break
        
        # Skip if we couldn't determine column
        if col_index is None:
            continue
        
        # First column is usually date
        if col_index == 0:
            date = parse_date(word_text)
            if date:
                transaction['Date'] = date
        
        # Last columns are usually amounts
        elif col_index >= len(x_positions) - 3:
            amount = parse_amount(word_text)
            if amount is not None:
                has_amounts = True
                
                # Rightmost column is usually balance
                if col_index == len(x_positions) - 1:
                    transaction['Balance'] = amount
                
                # Second-to-last is usually credit or debit
                elif col_index == len(x_positions) - 2:
                    if amount < 0:
                        transaction['Debit'] = abs(amount)
                    else:
                        transaction['Credit'] = amount
                    transaction['Amount'] = amount
                
                # Third-to-last could be the other of credit/debit
                elif col_index == len(x_positions) - 3:
                    if 'Amount' not in transaction or transaction['Amount'] is None:
                        if amount < 0:
                            transaction['Debit'] = abs(amount)
                        else:
                            transaction['Credit'] = amount
                        transaction['Amount'] = amount
        
        # Middle columns are usually description
        else:
            if transaction['Transaction Details']:
                transaction['Transaction Details'] += " " + word_text
            else:
                transaction['Transaction Details'] = word_text
    
    return has_amounts

def parse_date(text):
    """
    Parse a date string using various formats.
    
    Args:
        text (str): Date string
    
    Returns:
        datetime: Parsed date or None if parsing failed
    """
    if not text:
        return None
        
    text = str(text).strip()
    
    # Common date formats
    date_formats = [
        '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%m-%d-%Y',
        '%d/%m/%y', '%m/%d/%y', '%d-%m-%y', '%m-%d-%y',
        '%d %b %Y', '%d %B %Y', '%b %d %Y', '%B %d %Y'
    ]
    
    # Try exact match with formats
    for date_format in date_formats:
        try:
            return pd.to_datetime(text, format=date_format)
        except:
            pass
    
    # Try with pandas' flexible parser
    try:
        return pd.to_datetime(text)
    except:
        pass
    
    # Try to extract date with regex if it's part of a larger string
    date_patterns = [
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # DD/MM/YYYY or MM/DD/YYYY
        r'(\d{1,2} [A-Za-z]{3,} \d{2,4})'     # DD MMM YYYY
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return pd.to_datetime(match.group(1))
            except:
                pass
    
    return None

def parse_amount(text):
    """
    Parse an amount string, handling various formats.
    
    Args:
        text (str): Amount string
    
    Returns:
        float: Parsed amount or None if parsing failed
    """
    if text is None:
        return None
        
    # Convert to string
    text = str(text).strip()
    
    # If it's already a number, return it
    if isinstance(text, (int, float)):
        return float(text)
    
    # Remove currency symbols and commas
    text = re.sub(r'[$€£¥,]', '', text)
    
    # Check for debit/credit indicators
    debit_indicators = ['dr', 'debit', '-']
    credit_indicators = ['cr', 'credit', '+']
    
    is_debit = any(indicator in text.lower() for indicator in debit_indicators)
    is_credit = any(indicator in text.lower() for indicator in credit_indicators)
    
    # Remove indicators
    for indicator in debit_indicators + credit_indicators:
        text = text.lower().replace(indicator, '')
    
    # Extract numeric part
    match = re.search(r'-?\d+\.?\d*', text)
    if not match:
        return None
    
    try:
        amount = float(match.group(0))
        
        # Apply debit/credit
        if is_debit:
            amount = -abs(amount)
        elif is_credit:
            amount = abs(amount)
            
        return amount
    except:
        return None

def standardize_transaction_columns(df):
    """
    Standardize transaction DataFrame columns and types.
    
    Args:
        df (pandas.DataFrame): Transaction DataFrame
    
    Returns:
        pandas.DataFrame: Standardized DataFrame
    """
    # Ensure required columns exist
    required_columns = ['Date', 'Transaction Details', 'Debit', 'Credit', 'Amount', 'Balance']
    
    for col in required_columns:
        if col not in df.columns:
            if col == 'Date':
                df[col] = pd.to_datetime('today')
            elif col in ['Debit', 'Credit', 'Amount', 'Balance']:
                df[col] = 0.0
            else:
                df[col] = ""
    
    # Convert columns to proper types
    df['Date'] = pd.to_datetime(df['Date'])
    df['Time'] = df['Date'].astype(np.int64) // 10**9  # Add Time column for internal use
    
    for col in ['Debit', 'Credit', 'Amount', 'Balance']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    # Ensure Debit, Credit and Amount are consistent
    if 'Debit' in df.columns and 'Credit' in df.columns and 'Amount' in df.columns:
        # Recalculate Amount from Debit and Credit
        df['Amount'] = df['Credit'] - df['Debit']
    elif 'Amount' in df.columns:
        # Split Amount into Debit and Credit
        df['Debit'] = df['Amount'].apply(lambda x: abs(x) if x < 0 else 0)
        df['Credit'] = df['Amount'].apply(lambda x: x if x > 0 else 0)
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df

def convert_pdf_to_transactions(pdf_file):
    """
    Main function to convert a PDF bank statement to transaction DataFrame.
    
    Args:
        pdf_file (UploadedFile): Streamlit uploaded file
    
    Returns:
        pandas.DataFrame: Transaction DataFrame
    """
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(pdf_file.getvalue())
        temp_path = temp_file.name
    
    try:
        # Process the PDF
        st.info("Analyzing bank statement structure...")
        df = extract_transactions_from_bank_statement(temp_path)
        
        if not df.empty:
            st.success(f"Successfully extracted {len(df)} transactions from the bank statement.")
            return df
        else:
            st.error("No transactions could be extracted from the statement. Please try a different file.")
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

def display_extracted_transactions(df):
    """
    Display extracted transactions with edit options.
    
    Args:
        df (pandas.DataFrame): Transaction DataFrame
    
    Returns:
        pandas.DataFrame: Potentially modified DataFrame
    """
    st.subheader("Extracted Bank Transactions")
    
    # Format date for display
    display_df = df.copy()
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    
    # Show data in bank statement format
    display_columns = ['Date', 'Transaction Details', 'Debit', 'Credit', 'Balance']
    st.dataframe(display_df[display_columns])
    
    st.write(f"Found {len(df)} transactions. You can edit or add transactions below.")
    
    # Create tabs for different operations
    edit_tabs = st.tabs(["Edit/Delete Transactions", "Add Transaction", "Export Data"])
    
    with edit_tabs[0]:
        st.subheader("Edit or Delete Transactions")
        
        # Add filtering options
        if st.checkbox("Enable filtering"):
            filter_options = st.multiselect(
                "Filter by:",
                options=["Date Range", "Transaction Type", "Amount Range"],
                default=["Date Range"]
            )
            
            filtered_df = df.copy()
            
            # Date range filter
            if "Date Range" in filter_options and 'Date' in df.columns:
                min_date = df['Date'].min().date()
                max_date = df['Date'].max().date()
                date_range = st.date_input(
                    "Select date range:",
                    value=(min_date, max_date)
                )
                
                if len(date_range) == 2:
                    filtered_df = filtered_df[
                        (filtered_df['Date'].dt.date >= date_range[0]) &
                        (filtered_df['Date'].dt.date <= date_range[1])
                    ]
            
            # Transaction type filter
            if "Transaction Type" in filter_options:
                transaction_types = []
                if st.checkbox("Show deposits/credits", value=True):
                    transaction_types.append("Credit")
                if st.checkbox("Show withdrawals/debits", value=True):
                    transaction_types.append("Debit")
                
                if transaction_types:
                    if "Credit" in transaction_types and "Debit" not in transaction_types:
                        filtered_df = filtered_df[filtered_df['Credit'] > 0]
                    elif "Debit" in transaction_types and "Credit" not in transaction_types:
                        filtered_df = filtered_df[filtered_df['Debit'] > 0]
            
            # Amount range filter
            if "Amount Range" in filter_options:
                min_amount = float(df['Amount'].min())
                max_amount = float(df['Amount'].max())
                amount_range = st.slider(
                    "Filter by transaction amount:",
                    min_value=min_amount,
                    max_value=max_amount,
                    value=(min_amount, max_amount)
                )
                
                filtered_df = filtered_df[
                    (filtered_df['Amount'] >= amount_range[0]) &
                    (filtered_df['Amount'] <= amount_range[1])
                ]
            
            # Show filtered results
            st.write(f"Showing {len(filtered_df)} of {len(df)} transactions")
            
            # Format for display
            display_filtered = filtered_df.copy()
            display_filtered['Date'] = display_filtered['Date'].dt.strftime('%Y-%m-%d')
            
            st.dataframe(display_filtered[display_columns])
            
            # Delete option
            if st.button("Delete filtered transactions"):
                # Keep only rows not in filtered_df
                df = df.drop(filtered_df.index).reset_index(drop=True)
                st.success(f"Deleted {len(filtered_df)} transactions")
                
                # Show updated data
                display_df = df.copy()
                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                st.dataframe(display_df[display_columns])
    
    with edit_tabs[1]:
        st.subheader("Add New Transaction")
        
        with st.form("add_transaction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_date = st.date_input("Transaction Date", value=pd.to_datetime('today'))
                
                # Transaction type selection
                transaction_type = st.radio(
                    "Transaction Type",
                    ["Debit (Payment/Withdrawal)", "Credit (Deposit/Income)"]
                )
            
            with col2:
                # Amount (always positive, we'll apply sign based on type)
                amount = st.number_input("Amount (positive value)", 
                                        min_value=0.0, value=0.0, step=0.01, format="%.2f")
                
                # Balance after transaction
                balance = st.number_input("Balance After Transaction",
                                        value=0.0, step=0.01, format="%.2f")
            
            # Transaction details
            details = st.text_area("Transaction Details")
            
            submit_button = st.form_submit_button("Add Transaction")
            if submit_button:
                # Create new transaction
                new_transaction = {
                    'Date': pd.to_datetime(new_date),
                    'Time': pd.to_datetime(new_date).timestamp(),
                    'Transaction Details': details,
                    'Balance': balance
                }
                
                # Set debit/credit based on type
                if transaction_type.startswith("Debit"):
                    new_transaction['Debit'] = amount
                    new_transaction['Credit'] = 0.0
                    new_transaction['Amount'] = -amount  # Negative for debits
                else:  # Credit
                    new_transaction['Debit'] = 0.0
                    new_transaction['Credit'] = amount
                    new_transaction['Amount'] = amount  # Positive for credits
                
                # Add to DataFrame
                df = pd.concat([df, pd.DataFrame([new_transaction])], ignore_index=True)
                
                # Sort by date
                df = df.sort_values('Date').reset_index(drop=True)
                
                st.success("Transaction added successfully!")
                
                # Show updated data
                display_df = df.copy()
                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                st.dataframe(display_df[display_columns])
    
    with edit_tabs[2]:
        st.subheader("Export Transactions")
        
        export_format = st.radio(
            "Export Format",
            ["CSV", "Excel"]
        )
        
        if export_format == "CSV":
            csv = df.to_csv(index=False)
            st.download_button(
                "Download as CSV",
                data=csv,
                file_name="bank_transactions.csv",
                mime="text/csv"
            )
        else:
            # Create Excel file in memory
            buffer = io.BytesIO()
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
                money_format = workbook.add_format({'num_format': '$#,##0.00'})
                date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})
                
                # Apply formats to columns
                worksheet.set_column('A:A', 12, date_format)  # Date
                worksheet.set_column('C:F', 12, money_format)  # Money columns
                worksheet.set_column('B:B', 40)  # Transaction details
            
            # Get Excel data
            buffer.seek(0)
            
            st.download_button(
                "Download as Excel",
                data=buffer,
                file_name="bank_transactions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    return df
import pdfplumber
import tabula
import pandas as pd
import streamlit as st
import tempfile
import os
import re
import numpy as np
from datetime import datetime

def extract_text_with_pdfplumber(pdf_path):
    """
    Extract text and tables from PDF using pdfplumber
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        tuple: (text_content, tables)
    """
    text_content = []
    all_tables = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Extract text
                text = page.extract_text()
                if text:
                    text_content.append(text)
                
                # Try to extract tables with pdfplumber as well
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        # Convert table to pandas DataFrame
                        if table and len(table) > 1:  # Check for header and at least one row
                            headers = table[0]
                            data = table[1:]
                            # Create DataFrame with extracted data
                            df = pd.DataFrame(data, columns=headers)
                            all_tables.append(df)
    except Exception as e:
        st.error(f"Error extracting with pdfplumber: {str(e)}")
        
    return text_content, all_tables

def extract_tables_with_tabula(pdf_path):
    """
    Extract tables from PDF using tabula-py
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        list: List of pandas DataFrames containing tables
    """
    try:
        # Using lattice=True helps with extracting tables with visible borders
        tables_lattice = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True, lattice=True)
        
        # Using stream=True helps with extracting tables without visible borders
        tables_stream = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True, stream=True)
        
        # Try different areas of the pages - bank statements often have tables in specific regions
        tables_area = tabula.read_pdf(
            pdf_path, 
            pages='all', 
            multiple_tables=True, 
            area=[10, 10, 95, 95],  # Try to find tables in the center-ish of the page
            relative_area=True      # Use percentage of the page
        )
        
        # Combine all extracted tables
        all_tables = tables_lattice + tables_stream + tables_area
        
        # Remove empty tables
        all_tables = [table for table in all_tables if not table.empty]
        
        # Sort tables by number of rows (most likely the transaction table has the most rows)
        all_tables.sort(key=lambda x: len(x), reverse=True)
        
        return all_tables
    except Exception as e:
        st.error(f"Error extracting tables with tabula: {str(e)}")
        return []

def identify_date_columns(df):
    """
    Identify columns that contain date information
    
    Args:
        df (pandas.DataFrame): DataFrame to analyze
        
    Returns:
        list: List of column names that likely contain dates
    """
    date_cols = []
    date_patterns = [
        r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY or DD/MM/YYYY
        r'\d{1,2}-\d{1,2}-\d{2,4}',  # MM-DD-YYYY or DD-MM-YYYY
        r'\d{4}-\d{1,2}-\d{1,2}',    # YYYY-MM-DD
        r'\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}'  # DD Month YYYY or Month DD YYYY
    ]
    
    for col in df.columns:
        # Skip if column is not string type
        if df[col].dtype != 'object':
            continue
            
        # Count date pattern matches in the column
        matches = 0
        for pattern in date_patterns:
            for value in df[col].dropna():
                if re.search(pattern, str(value)):
                    matches += 1
                    break  # One match in this row is enough
        
        # If more than 50% of non-null values match a date pattern, mark as date column
        if matches > df[col].count() * 0.2:  # 20% threshold
            date_cols.append(col)
            
    return date_cols

def identify_amount_columns(df):
    """
    Identify columns that contain monetary amounts
    
    Args:
        df (pandas.DataFrame): DataFrame to analyze
        
    Returns:
        list: List of column names that likely contain amounts
    """
    amount_cols = []
    amount_patterns = [
        r'[\$€£¥]\s*\d+[,\d]*\.\d+',  # Currency symbol followed by number with decimal
        r'\d+[,\d]*\.\d+\s*[\$€£¥]',  # Number with decimal followed by currency symbol
        r'[\-\+]?\d+[,\d]*\.\d{2}'    # Positive or negative number with 2 decimal places
    ]
    
    for col in df.columns:
        # If column is numeric, it might be an amount
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if values have 2 decimal places which is common for currency
            decimal_count = df[col].astype(str).str.extract(r'\.(\d+)')[0].str.len().mean()
            if 1.5 < decimal_count < 2.5:  # Around 2 decimal places
                amount_cols.append(col)
            continue
            
        # For string columns, check for currency patterns
        if df[col].dtype == 'object':
            matches = 0
            for pattern in amount_patterns:
                for value in df[col].dropna():
                    if re.search(pattern, str(value)):
                        matches += 1
                        break  # One match in this row is enough
            
            # If more than 30% of non-null values match an amount pattern, mark as amount column
            if matches > df[col].count() * 0.3:  # 30% threshold
                amount_cols.append(col)
    
    # Sort by how likely it is to be an amount column
    # More negative values typically indicates an expense (debit) column
    column_scores = {}
    for col in amount_cols:
        try:
            # Convert to numeric, handling currency symbols and commas
            values = pd.to_numeric(
                df[col].astype(str)
                .str.replace(r'[\$€£¥,]', '', regex=True)
                .str.replace(r'[^-+.0-9]', '', regex=True), 
                errors='coerce'
            )
            # Calculate score based on number of negative values and presence of decimals
            neg_ratio = (values < 0).mean() if not values.isna().all() else 0
            has_decimals = values.astype(str).str.contains(r'\.').mean() if not values.isna().all() else 0
            column_scores[col] = neg_ratio * 2 + has_decimals
        except:
            column_scores[col] = 0
    
    # Sort amount columns by score (higher score = more likely to be transaction amount)
    amount_cols = sorted(amount_cols, key=lambda x: column_scores.get(x, 0), reverse=True)
    
    return amount_cols

def identify_description_columns(df):
    """
    Identify columns that likely contain transaction descriptions
    
    Args:
        df (pandas.DataFrame): DataFrame to analyze
        
    Returns:
        list: List of column names that likely contain descriptions
    """
    desc_cols = []
    
    for col in df.columns:
        # Skip non-string columns
        if df[col].dtype != 'object':
            continue
        
        # Check average string length
        avg_length = df[col].astype(str).str.len().mean()
        
        # Check word count
        avg_word_count = df[col].astype(str).str.split().str.len().mean()
        
        # Descriptions typically have longer text and multiple words
        if avg_length > 15 and avg_word_count > 2:
            desc_cols.append((col, avg_length))
    
    # Sort by average length (longer = more likely to be description)
    desc_cols.sort(key=lambda x: x[1], reverse=True)
    
    return [col for col, _ in desc_cols]

def parse_amount_string(amount_str):
    """
    Parse amount string to float, handling different formats
    
    Args:
        amount_str (str): String containing an amount
        
    Returns:
        float: Parsed amount
    """
    if pd.isna(amount_str) or amount_str is None:
        return 0.0
        
    amount_str = str(amount_str).strip()
    
    # Remove currency symbols and spaces
    amount_str = re.sub(r'[\$€£¥\s,]', '', amount_str)
    
    # Handle parentheses indicating negative numbers - (100.00) → -100.00
    if amount_str.startswith('(') and amount_str.endswith(')'):
        amount_str = '-' + amount_str[1:-1]
    
    # Handle CR/DR indicators
    if amount_str.upper().endswith('DR'):
        amount_str = '-' + amount_str[:-2]
    elif amount_str.upper().endswith('CR'):
        amount_str = amount_str[:-2]
    
    # Try to extract just the numbers and decimal point
    amount_match = re.search(r'([-+]?\d*\.?\d+)', amount_str)
    if amount_match:
        amount_str = amount_match.group(1)
    
    try:
        return float(amount_str)
    except ValueError:
        return 0.0

def parse_date_string(date_str, try_formats=None):
    """
    Parse date string to datetime, handling different formats
    
    Args:
        date_str (str): String containing a date
        try_formats (list): List of formats to try
        
    Returns:
        datetime: Parsed date
    """
    if pd.isna(date_str) or date_str is None:
        return pd.NaT
    
    date_str = str(date_str).strip()
    
    # Default formats to try
    if try_formats is None:
        try_formats = [
            '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d',  # Slash formats
            '%m-%d-%Y', '%d-%m-%Y', '%Y-%m-%d',  # Dash formats
            '%b %d, %Y', '%B %d, %Y',            # Month name formats
            '%d %b %Y', '%d %B %Y',              # Day first formats
            '%m/%d/%y', '%d/%m/%y',              # Two-digit year formats
            '%Y%m%d',                            # Compact format
        ]
    
    # Try each format
    for fmt in try_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # If direct parsing fails, try to extract a date pattern
    date_patterns = [
        r'(\d{1,2}/\d{1,2}/\d{2,4})',  # MM/DD/YYYY or DD/MM/YYYY
        r'(\d{1,2}-\d{1,2}-\d{2,4})',  # MM-DD-YYYY or DD-MM-YYYY
        r'(\d{4}-\d{1,2}-\d{1,2})',    # YYYY-MM-DD
        r'(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4})'  # DD Month YYYY or Month DD YYYY
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, date_str)
        if match:
            return parse_date_string(match.group(1), try_formats)
    
    return pd.NaT

def clean_dataframe(df):
    """
    Clean and process the extracted DataFrame
    
    Args:
        df (pandas.DataFrame): DataFrame to clean
        
    Returns:
        pandas.DataFrame: Cleaned DataFrame with standard columns for banking transactions
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Drop completely empty columns and rows
    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='all')
    
    # Identify different column types
    date_cols = identify_date_columns(df)
    amount_cols = identify_amount_columns(df)
    desc_cols = identify_description_columns(df)
    
    # Create a new DataFrame with standardized columns
    transactions = pd.DataFrame()
    
    # Process Date/Time column
    if date_cols:
        date_col = date_cols[0]  # Use the first identified date column
        transactions['Date'] = df[date_col].apply(parse_date_string)
        transactions['Time'] = transactions['Date'].astype(np.int64) // 10**9  # Convert to Unix timestamp
    else:
        # If no date column found, create a default
        st.warning("No date column identified. Using current date.")
        transactions['Date'] = pd.to_datetime('today')
        transactions['Time'] = transactions['Date'].astype(np.int64) // 10**9
    
    # Separate Credits and Debits instead of just Amount
    if len(amount_cols) > 1:
        # If we have multiple amount columns, try to identify which is credit and which is debit
        debit_col = None
        credit_col = None
        
        # Check column names for obvious indicators
        for col in amount_cols:
            col_lower = col.lower() if isinstance(col, str) else ""
            if any(term in col_lower for term in ['debit', 'dr', 'withdrawal', 'payment', 'out']):
                debit_col = col
            elif any(term in col_lower for term in ['credit', 'cr', 'deposit', 'in']):
                credit_col = col
        
        # If we still don't have both, try to determine based on values
        if not (debit_col and credit_col):
            # Calculate negative value frequency
            neg_frequencies = {}
            for col in amount_cols:
                try:
                    values = pd.to_numeric(
                        df[col].astype(str)
                        .str.replace(r'[\$€£¥,]', '', regex=True)
                        .str.replace(r'[^-+.0-9]', '', regex=True), 
                        errors='coerce'
                    )
                    neg_frequencies[col] = (values < 0).mean() if not values.isna().all() else 0
                except:
                    neg_frequencies[col] = 0
            
            # Sort by negative frequency - more negative values suggests debit
            sorted_cols = sorted(neg_frequencies.items(), key=lambda x: x[1], reverse=True)
            
            if len(sorted_cols) >= 2:
                # First column has most negative values, likely debit
                if not debit_col:
                    debit_col = sorted_cols[0][0]
                # Second column has fewer negative values, likely credit
                if not credit_col:
                    credit_col = sorted_cols[1][0]
            elif len(sorted_cols) == 1:
                # We only have one column - determine if it's mostly positive or negative
                col, neg_freq = sorted_cols[0]
                if neg_freq > 0.5:  # Mostly negative
                    debit_col = col
                else:  # Mostly positive
                    credit_col = col
        
        # Process Debit column (make values positive)
        if debit_col:
            values = df[debit_col].apply(parse_amount_string)
            transactions['Debit'] = values.abs()  # Make all values positive
        else:
            transactions['Debit'] = 0.0
            
        # Process Credit column
        if credit_col:
            transactions['Credit'] = df[credit_col].apply(parse_amount_string).abs()  # Make all values positive
        else:
            transactions['Credit'] = 0.0
            
        # Calculate Amount as Credit - Debit
        transactions['Amount'] = transactions['Credit'] - transactions['Debit']
        
    else:
        # If only one amount column, try to determine if positive values are credits and negative are debits
        if amount_cols:
            amount_col = amount_cols[0]
            values = df[amount_col].apply(parse_amount_string)
            
            # Split into debit and credit based on sign
            transactions['Debit'] = values.apply(lambda x: abs(x) if x < 0 else 0)
            transactions['Credit'] = values.apply(lambda x: x if x > 0 else 0)
            transactions['Amount'] = values
        else:
            # No amount columns found, try to find numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if not numeric_cols.empty:
                values = df[numeric_cols[0]]
                transactions['Debit'] = values.apply(lambda x: abs(x) if x < 0 else 0)
                transactions['Credit'] = values.apply(lambda x: x if x > 0 else 0)
                transactions['Amount'] = values
            else:
                st.warning("No amount columns identified. Using zeros.")
                transactions['Debit'] = 0.0
                transactions['Credit'] = 0.0
                transactions['Amount'] = 0.0
    
    # Transaction Details (Description) column
    if desc_cols:
        # Use the first identified description column
        desc_col = desc_cols[0]
        transactions['Transaction Details'] = df[desc_col]
    elif len(df.columns) > 2:
        # If no description column identified but we have other columns, use the first non-date, non-amount column
        other_cols = [col for col in df.columns if col not in date_cols + amount_cols]
        if other_cols:
            transactions['Transaction Details'] = df[other_cols[0]]
    else:
        transactions['Transaction Details'] = "Transaction"
    
    # Balance column
    balance_col = None
    for col in df.columns:
        col_lower = str(col).lower()
        if 'balance' in col_lower or ('bal' in col_lower and len(col_lower) < 8):
            balance_col = col
            break
    
    if balance_col:
        transactions['Balance'] = df[balance_col].apply(parse_amount_string)
    elif 'Balance' in df.columns:
        transactions['Balance'] = df['Balance'].apply(parse_amount_string)
    else:
        # Calculate a running balance if not found in the data
        if 'Amount' in transactions.columns:
            transactions['Balance'] = transactions['Amount'].cumsum()
    
    # Keep original type if available
    if 'Type' in df.columns:
        transactions['Type'] = df['Type']
    elif transactions['Amount'].any():
        # Create a derived type
        transactions['Type'] = transactions['Amount'].apply(
            lambda x: 'DEPOSIT' if x > 0 else 'PAYMENT'
        )
        
    # Add category if available
    if 'Category' in df.columns:
        transactions['Category'] = df['Category']
    
    # Ensure we have all the standard columns
    standard_columns = ['Date', 'Time', 'Transaction Details', 'Debit', 'Credit', 'Amount', 'Balance']
    for col in standard_columns:
        if col not in transactions.columns:
            if col in ['Debit', 'Credit', 'Amount', 'Balance']:
                transactions[col] = 0.0
            elif col in ['Date', 'Time']:
                transactions[col] = pd.to_datetime('today') if col == 'Date' else pd.to_datetime('today').timestamp()
            else:
                transactions[col] = ""
    
    # Ensure transactions are sorted by date
    if 'Date' in transactions.columns:
        transactions = transactions.sort_values('Date').reset_index(drop=True)
    
    return transactions

def analyze_pdf_structure(pdf_path):
    """
    Analyze the structure of the PDF to determine the best extraction method
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        dict: Analysis results with extraction strategy
    """
    result = {
        'has_tables': False,
        'has_text': False,
        'recommended_method': 'unknown',
        'page_count': 0
    }
    
    try:
        # Check if tabula can detect tables
        tables = tabula.read_pdf(pdf_path, pages='1', multiple_tables=True)
        result['has_tables'] = len(tables) > 0 and any(not table.empty for table in tables)
        
        # Check if pdfplumber can extract text
        with pdfplumber.open(pdf_path) as pdf:
            result['page_count'] = len(pdf.pages)
            first_page_text = pdf.pages[0].extract_text() if pdf.pages else ""
            result['has_text'] = bool(first_page_text)
            
            # Count potential table rows by looking for common transaction indicators
            transaction_indicators = [
                r'\d{1,2}/\d{1,2}',  # Date pattern like MM/DD or DD/MM
                r'\$\d+\.\d{2}',     # Dollar amount
                r'€\d+\.\d{2}',      # Euro amount
                r'£\d+\.\d{2}',      # Pound amount
                r'DEPOSIT',           # Common transaction words
                r'WITHDRAWAL',
                r'PURCHASE',
                r'PAYMENT'
            ]
            
            indicator_matches = 0
            for line in first_page_text.split('\n'):
                for indicator in transaction_indicators:
                    if re.search(indicator, line):
                        indicator_matches += 1
                        break
            
            result['transaction_indicator_count'] = indicator_matches
        
        # Determine recommended extraction method
        if result['has_tables']:
            result['recommended_method'] = 'tabula'
        elif result['has_text'] and result['transaction_indicator_count'] > 3:
            result['recommended_method'] = 'pdfplumber_text'
        elif result['has_text']:
            result['recommended_method'] = 'hybrid'
        else:
            result['recommended_method'] = 'ocr'  # Not implemented but could be added
            
    except Exception as e:
        st.error(f"Error analyzing PDF structure: {str(e)}")
        
    return result

def extract_transactions_from_text(text_content):
    """
    Extract transaction data from plain text using pattern matching
    
    Args:
        text_content (list): List of text content from each page
        
    Returns:
        pandas.DataFrame: DataFrame containing the extracted transactions
    """
    transactions = []
    
    # Common patterns
    date_pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
    amount_pattern = r'([-+$€£]?\s?\d{1,3}(?:,\d{3})*\.\d{2})'
    balance_pattern = r'(?:balance|bal)[\s:]+([0-9,.]+)'
    
    # Process each page
    for page_text in text_content:
        lines = page_text.split('\n')
        running_balance = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip short lines as they're unlikely to be transactions
            if len(line) < 10:
                continue
            
            # Check for balance information
            balance_match = re.search(balance_pattern, line.lower())
            if balance_match:
                try:
                    running_balance = parse_amount_string(balance_match.group(1))
                except:
                    pass
                
            # Look for date and amount patterns
            date_matches = list(re.finditer(date_pattern, line))
            amount_matches = list(re.finditer(amount_pattern, line))
            
            # If we have both date and amount, treat as a transaction line
            if date_matches and amount_matches:
                # Get the first date match
                date_match = date_matches[0]
                date_str = date_match.group(1)
                date_end = date_match.end()
                
                # Try to detect credit vs debit
                # First, check if there are multiple amount matches (potential separate credit/debit columns)
                debit_amount = 0.0
                credit_amount = 0.0
                description = ""
                
                if len(amount_matches) >= 2:
                    # Multiple amounts - try to determine which is credit and which is debit
                    amounts = [parse_amount_string(m.group(1)) for m in amount_matches]
                    
                    # Check if any amount is negative
                    neg_amounts = [a for a in amounts if a < 0]
                    pos_amounts = [a for a in amounts if a > 0]
                    
                    if neg_amounts:
                        # Treat negative values as debits (make positive)
                        debit_amount = abs(neg_amounts[0])
                        # If we also have positive, use that as credit
                        if pos_amounts:
                            credit_amount = pos_amounts[0]
                    elif len(amounts) >= 2:
                        # If both are positive, assume first is debit, second is credit
                        # This is a simplification and may need adjustment for specific banks
                        debit_amount = amounts[0]
                        credit_amount = amounts[1]
                    else:
                        # Only one positive amount, determine if it's credit or debit based on context
                        amount = amounts[0]
                        if "deposit" in line.lower() or "credit" in line.lower():
                            credit_amount = amount
                        else:
                            debit_amount = amount
                    
                    # Extract description - everything between date and first amount
                    first_amount_start = min(m.start() for m in amount_matches)
                    if date_end < first_amount_start:
                        description = line[date_end:first_amount_start].strip()
                    else:
                        description = line[date_end:].strip()
                
                else:
                    # Just one amount - determine if it's credit or debit
                    amount_str = amount_matches[0].group(1)
                    amount = parse_amount_string(amount_str)
                    
                    # Extract description (everything between date and amount or after date)
                    amount_start = amount_matches[0].start()
                    
                    if date_end < amount_start:
                        description = line[date_end:amount_start].strip()
                    else:
                        description = line[date_end:].strip()
                        if not description and i+1 < len(lines):
                            description = lines[i+1].strip()
                    
                    # Try to determine if credit or debit based on context words in description
                    debit_words = ['payment', 'purchase', 'withdrawal', 'debit', 'paid', 'fee', 'transfer to']
                    credit_words = ['deposit', 'credit', 'refund', 'transfer from', 'interest', 'salary']
                    
                    if amount < 0:
                        # Negative amount is always debit
                        debit_amount = abs(amount)
                    elif any(word in description.lower() for word in debit_words):
                        debit_amount = amount
                    elif any(word in description.lower() for word in credit_words):
                        credit_amount = amount
                    else:
                        # If we can't determine, use the sign
                        if amount >= 0:
                            credit_amount = amount
                        else:
                            debit_amount = abs(amount)
                
                # Clean up description
                description = re.sub(r'\s+', ' ', description)
                
                # Calculate amount (Credit - Debit)
                net_amount = credit_amount - debit_amount
                
                # Try to detect balance information
                balance = None
                # Look for a balance pattern at the end of the line
                if running_balance is not None:
                    balance = running_balance
                else:
                    # If multiple amounts and the last doesn't look like credit/debit, might be balance
                    if len(amount_matches) >= 3:
                        balance_candidate = parse_amount_string(amount_matches[-1].group(1))
                        # If balance is much larger than transaction amount, likely a balance
                        if abs(balance_candidate) > abs(net_amount) * 5:
                            balance = balance_candidate
                
                # Add to transactions
                transaction = {
                    'Date': parse_date_string(date_str),
                    'Transaction Details': description,
                    'Debit': debit_amount,
                    'Credit': credit_amount,
                    'Amount': net_amount
                }
                
                if balance is not None:
                    transaction['Balance'] = balance
                
                transactions.append(transaction)
    
    if transactions:
        df = pd.DataFrame(transactions)
        df['Time'] = df['Date'].astype(np.int64) // 10**9  # Convert to Unix timestamp
        
        # If no balance column but we have amounts, calculate running balance
        if 'Balance' not in df.columns and 'Amount' in df.columns:
            df['Balance'] = df['Amount'].cumsum()
        
        return df
    else:
        return None

def convert_pdf_to_csv(pdf_file):
    """
    Convert a PDF bank statement to CSV format using multiple strategies
    
    Args:
        pdf_file (UploadedFile): Streamlit uploaded file object
        
    Returns:
        pandas.DataFrame: DataFrame containing the extracted data
    """
    # Create a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(pdf_file.getvalue())
        temp_path = temp_file.name
    
    try:
        # Analyze PDF structure to determine best extraction method
        st.info("Analyzing PDF structure...")
        pdf_analysis = analyze_pdf_structure(temp_path)
        
        st.info(f"PDF has {pdf_analysis['page_count']} pages. " +
                f"Contains tables: {'Yes' if pdf_analysis['has_tables'] else 'No'}. " +
                f"Contains text: {'Yes' if pdf_analysis['has_text'] else 'No'}. " +
                f"Recommended extraction method: {pdf_analysis['recommended_method']}.")
        
        # Extract data using the recommended method
        if pdf_analysis['recommended_method'] == 'tabula':
            st.info("Extracting tables from PDF...")
            tables = extract_tables_with_tabula(temp_path)
            
            if tables:
                # If multiple tables were found, show user and let them choose
                if len(tables) > 1:
                    st.success(f"Found {len(tables)} tables in the PDF.")
                    
                    # Combine all tables that are likely to be transaction tables
                    processed_tables = []
                    for table in tables:
                        if len(table) > 3:  # Skip very small tables
                            processed_table = clean_dataframe(table)
                            if not processed_table.empty:
                                processed_tables.append(processed_table)
                    
                    if processed_tables:
                        # Combine all processed tables
                        combined_df = pd.concat(processed_tables)
                        return combined_df
                    else:
                        st.warning("No usable transaction tables found after processing.")
                else:
                    # Just one table found, process it
                    df = clean_dataframe(tables[0])
                    return df
        
        # If tabula didn't work or wasn't recommended, try pdfplumber
        st.info("Extracting text and tables with pdfplumber...")
        text_content, plumber_tables = extract_text_with_pdfplumber(temp_path)
        
        # Process pdfplumber tables if found
        if plumber_tables:
            st.success(f"Found {len(plumber_tables)} tables with pdfplumber.")
            
            processed_tables = []
            for table in plumber_tables:
                if len(table) > 3:  # Skip very small tables
                    processed_table = clean_dataframe(table)
                    if not processed_table.empty and 'Time' in processed_table.columns:
                        processed_tables.append(processed_table)
            
            if processed_tables:
                # Combine all processed tables
                combined_df = pd.concat(processed_tables)
                return combined_df
        
        # If no tables found or processing failed, try text-based extraction
        if text_content:
            st.info("Attempting to extract transactions from text...")
            df = extract_transactions_from_text(text_content)
            if df is not None and not df.empty:
                st.success("Successfully extracted transactions from text content.")
                return df
        
        # If all methods failed, create a minimal DataFrame for manual entry
        st.warning("Automated extraction methods didn't find transaction data. Creating blank template.")
        return pd.DataFrame({
            'Date': [pd.to_datetime('today')],
            'Time': [pd.to_datetime('today').timestamp()],
            'Amount': [0.0],
            'Description': ['Enter transaction details manually']
        })
            
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        # Return a minimal DataFrame as fallback
        return pd.DataFrame({
            'Date': [pd.to_datetime('today')],
            'Time': [pd.to_datetime('today').timestamp()],
            'Amount': [0.0],
            'Description': ['Error processing PDF']
        })
    finally:
        # Clean up the temporary file
        os.unlink(temp_path)

def display_extracted_data(df):
    """
    Display extracted data and allow users to adjust it
    
    Args:
        df (pandas.DataFrame): DataFrame to display
        
    Returns:
        pandas.DataFrame: Potentially modified DataFrame
    """
    st.subheader("Extracted Transaction Data")
    
    # Ensure we have the standard columns with proper defaults
    standard_columns = ['Date', 'Transaction Details', 'Debit', 'Credit', 'Balance']
    for col in standard_columns:
        if col not in df.columns:
            if col == 'Date':
                df[col] = pd.to_datetime('today')
            elif col in ['Debit', 'Credit', 'Balance']:
                df[col] = 0.0
            else:
                df[col] = ""
    
    # Format the date for display
    display_df = df.copy()
    if 'Date' in display_df.columns:
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    
    # Show a clean display of the data with bank statement format
    display_columns = [col for col in standard_columns if col in display_df.columns]
    st.dataframe(display_df[display_columns])
    
    st.write("If the data doesn't look correct, you can adjust it below.")
    
    # Create tabs for different adjustment options
    adjustment_tabs = st.tabs(["Column Mapping", "Edit Transactions", "Add Transaction"])
    
    with adjustment_tabs[0]:
        st.subheader("Map Columns")
        
        # Show available columns
        st.write("Available columns: " + ", ".join(df.columns))
        
        col_map = {}
        
        # Create column mapping UI
        col1, col2 = st.columns(2)
        
        with col1:
            # Date mapping
            if 'Date' in df.columns:
                st.write("✅ Date column found")
            else:
                date_options = [col for col in df.columns if 'date' in col.lower()]
                date_index = 0
                if date_options:
                    date_index = list(df.columns).index(date_options[0])
                
                date_col = st.selectbox(
                    "Select the column containing dates:",
                    options=df.columns,
                    index=date_index
                )
                col_map['Date'] = date_col
            
            # Debit mapping
            if 'Debit' in df.columns:
                st.write("✅ Debit column found")
            else:
                debit_options = [col for col in df.columns 
                              if any(term in col.lower() for term in ['debit', 'payment', 'dr', 'withdraw', 'out'])]
                debit_index = 0
                if debit_options:
                    debit_index = list(df.columns).index(debit_options[0])
                elif 'Amount' in df.columns:  # Use Amount as fallback
                    debit_index = list(df.columns).index('Amount')
                
                debit_col = st.selectbox(
                    "Select the column containing debits/payments:",
                    options=df.columns,
                    index=debit_index
                )
                col_map['Debit'] = debit_col
        
        with col2:
            # Credit mapping
            if 'Credit' in df.columns:
                st.write("✅ Credit column found")
            else:
                credit_options = [col for col in df.columns 
                               if any(term in col.lower() for term in ['credit', 'deposit', 'cr', 'in'])]
                credit_index = 0
                if credit_options:
                    credit_index = list(df.columns).index(credit_options[0])
                elif 'Amount' in df.columns and 'Debit' not in df.columns:  # If already using Amount for debit, don't use again
                    credit_index = list(df.columns).index('Amount')
                
                credit_col = st.selectbox(
                    "Select the column containing credits/deposits:",
                    options=df.columns,
                    index=credit_index
                )
                col_map['Credit'] = credit_col
            
            # Balance mapping  
            if 'Balance' in df.columns:
                st.write("✅ Balance column found")
            else:
                balance_options = [col for col in df.columns 
                                if any(term in col.lower() for term in ['balance', 'bal', 'ending'])]
                balance_index = 0
                if balance_options:
                    balance_index = list(df.columns).index(balance_options[0])
                
                # Only show if we found a likely balance column
                if balance_options:
                    balance_col = st.selectbox(
                        "Select the column containing balance:",
                        options=df.columns,
                        index=balance_index
                    )
                    col_map['Balance'] = balance_col
        
        # Transaction Details mapping
        if 'Transaction Details' not in df.columns:
            if 'Description' in df.columns:
                col_map['Transaction Details'] = 'Description'
            else:
                desc_options = [col for col in df.columns 
                             if any(term in col.lower() for term in ['description', 'details', 'narration', 'transaction', 'particulars'])]
                if desc_options:
                    desc_index = list(df.columns).index(desc_options[0])
                    desc_col = st.selectbox(
                        "Select the column containing transaction details:",
                        options=df.columns,
                        index=desc_index
                    )
                    col_map['Transaction Details'] = desc_col
        
        # Apply column mappings if needed
        if col_map:
            if st.button("Apply Column Mapping"):
                # Create a new DataFrame with mapped columns
                mapped_df = df.copy()
                
                # Add mapped columns
                for new_col, old_col in col_map.items():
                    if old_col in df.columns:
                        mapped_df[new_col] = df[old_col]
                
                # Ensure Time column exists for internal processing
                if 'Date' in mapped_df.columns and 'Time' not in mapped_df.columns:
                    mapped_df['Time'] = pd.to_datetime(mapped_df['Date']).astype(np.int64) // 10**9
                
                # If we have Amount but not separate Debit/Credit, try to split them
                if 'Amount' in mapped_df.columns and (
                    ('Debit' not in mapped_df.columns) or ('Credit' not in mapped_df.columns)
                ):
                    # Split Amount into Debit/Credit based on sign
                    if 'Debit' not in mapped_df.columns:
                        mapped_df['Debit'] = mapped_df['Amount'].apply(lambda x: abs(x) if x < 0 else 0)
                    
                    if 'Credit' not in mapped_df.columns:
                        mapped_df['Credit'] = mapped_df['Amount'].apply(lambda x: x if x > 0 else 0)
                
                # If we have Debit/Credit but not Amount, calculate it
                elif 'Debit' in mapped_df.columns and 'Credit' in mapped_df.columns and 'Amount' not in mapped_df.columns:
                    mapped_df['Amount'] = mapped_df['Credit'] - mapped_df['Debit']
                
                df = mapped_df
                st.success("Column mapping applied!")
                
                # Format the date for display
                display_df = df.copy()
                if 'Date' in display_df.columns:
                    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                
                # Show updated data in bank statement format
                display_columns = [col for col in standard_columns if col in display_df.columns]
                st.dataframe(display_df[display_columns])
    
    with adjustment_tabs[1]:
        st.subheader("Edit Transactions")
        
        # Allow for filtering or deleting rows
        if st.checkbox("Enable row filtering/deletion"):
            # Create multi-column filters
            filter_options = st.multiselect(
                "Filter by:",
                options=["Date Range", "Debit Amount", "Credit Amount", "Transaction Type"],
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
            
            # Debit amount filter
            if "Debit Amount" in filter_options and 'Debit' in df.columns:
                min_debit = float(df['Debit'].min())
                max_debit = float(df['Debit'].max()) if max_debit > min_debit else min_debit + 100
                debit_range = st.slider(
                    "Filter by debit amount:",
                    min_value=min_debit,
                    max_value=max_debit,
                    value=(min_debit, max_debit)
                )
                
                filtered_df = filtered_df[
                    (filtered_df['Debit'] >= debit_range[0]) &
                    (filtered_df['Debit'] <= debit_range[1])
                ]
            
            # Credit amount filter
            if "Credit Amount" in filter_options and 'Credit' in df.columns:
                min_credit = float(df['Credit'].min())
                max_credit = float(df['Credit'].max()) if max_credit > min_credit else min_credit + 100
                credit_range = st.slider(
                    "Filter by credit amount:",
                    min_value=min_credit,
                    max_value=max_credit,
                    value=(min_credit, max_credit)
                )
                
                filtered_df = filtered_df[
                    (filtered_df['Credit'] >= credit_range[0]) &
                    (filtered_df['Credit'] <= credit_range[1])
                ]
            
            # Transaction type filter 
            if "Transaction Type" in filter_options:
                if 'Debit' in df.columns and 'Credit' in df.columns:
                    transaction_types = []
                    if st.checkbox("Show deposits/credits", value=True):
                        transaction_types.append("Credit")
                    if st.checkbox("Show withdrawals/debits", value=True):
                        transaction_types.append("Debit")
                    
                    if "Credit" in transaction_types and "Debit" not in transaction_types:
                        filtered_df = filtered_df[filtered_df['Credit'] > 0]
                    elif "Debit" in transaction_types and "Credit" not in transaction_types:
                        filtered_df = filtered_df[filtered_df['Debit'] > 0]
            
            # Show filtered data
            st.write(f"Showing {len(filtered_df)} of {len(df)} transactions")
            
            # Format the date for display
            display_filtered = filtered_df.copy()
            if 'Date' in display_filtered.columns:
                display_filtered['Date'] = display_filtered['Date'].dt.strftime('%Y-%m-%d')
            
            # Display in bank statement format  
            display_columns = [col for col in standard_columns if col in display_filtered.columns]
            st.dataframe(display_filtered[display_columns])
            
            # Delete filtered rows
            if st.button("Delete filtered transactions"):
                # Get indices to delete
                indices_to_delete = filtered_df.index
                df = df.drop(indices_to_delete).reset_index(drop=True)
                st.success(f"Deleted {len(indices_to_delete)} transactions")
                
                # Display updated data
                display_df = df.copy()
                if 'Date' in display_df.columns:
                    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                st.dataframe(display_df[display_columns])
    
    with adjustment_tabs[2]:
        st.subheader("Add Transaction")
        
        # Add a new transaction with separate debit/credit
        with st.form("add_transaction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_date = st.date_input("Date", value=pd.to_datetime('today'))
                
                # Use radio to select if it's a credit or debit
                transaction_type = st.radio(
                    "Transaction Type",
                    ["Debit (Payment/Withdrawal)", "Credit (Deposit/Income)"]
                )
            
            with col2:
                # Just ask for a positive amount that we'll interpret based on type
                new_amount = st.number_input("Amount (positive value)", 
                                           min_value=0.0, value=0.0, format="%.2f")
                
                # Optional balance field
                new_balance = st.number_input("Balance (optional)", 
                                            value=0.0, format="%.2f")
            
            # Transaction details
            new_details = st.text_input("Transaction Details")
            
            submit_button = st.form_submit_button("Add Transaction")
            if submit_button:
                # Create the transaction based on the type
                new_row = {
                    'Date': pd.to_datetime(new_date),
                    'Time': pd.to_datetime(new_date).timestamp(),
                    'Transaction Details': new_details,
                    'Balance': new_balance
                }
                
                # Set debit/credit based on transaction type
                if transaction_type.startswith("Debit"):
                    new_row['Debit'] = new_amount
                    new_row['Credit'] = 0.0
                    new_row['Amount'] = -new_amount  # Negative for debits
                else:  # Credit
                    new_row['Debit'] = 0.0
                    new_row['Credit'] = new_amount
                    new_row['Amount'] = new_amount  # Positive for credits
                
                # Add any additional columns from original df with empty values
                for col in df.columns:
                    if col not in new_row:
                        new_row[col] = None
                
                # Append the new row
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                
                # Sort by date
                if 'Date' in df.columns:
                    df = df.sort_values('Date').reset_index(drop=True)
                
                st.success("Transaction added!")
                
                # Display updated data
                display_df = df.copy()
                if 'Date' in display_df.columns:
                    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                
                display_columns = [col for col in standard_columns if col in display_df.columns]
                st.dataframe(display_df[display_columns])
    
    # Final data format check
    if 'Time' not in df.columns and 'Date' in df.columns:
        df['Time'] = pd.to_datetime(df['Date']).astype(np.int64) // 10**9
    
    if 'Amount' not in df.columns:
        # Try to calculate Amount from Debit/Credit
        if 'Debit' in df.columns and 'Credit' in df.columns:
            df['Amount'] = df['Credit'] - df['Debit']
        else:
            st.error("Amount column is required. Please map or add it.")
            df['Amount'] = 0.0
    
    # Convert columns to the right types
    if 'Time' in df.columns and df['Time'].dtype != 'float64':
        try:
            df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
        except:
            st.warning("Could not convert Time to numeric format. Using timestamps.")
            df['Time'] = range(len(df))
    
    if 'Amount' in df.columns and df['Amount'].dtype != 'float64':
        try:
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        except:
            st.warning("Could not convert Amount to numeric format. Using zeros.")
            df['Amount'] = 0.0
    
    # Show final data for analysis
    st.subheader("Final Data for Analysis")
    final_display = df.copy()
    
    # Format date for display
    if 'Date' in final_display.columns:
        final_display['Date'] = final_display['Date'].dt.strftime('%Y-%m-%d')
    
    # Calculate display columns - prioritize the bank statement format
    display_columns = [col for col in standard_columns if col in final_display.columns]
    st.dataframe(final_display[display_columns])
    
    # Export options
    st.subheader("Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download processed data
        csv = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="extracted_transactions.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download raw data for debugging
        if st.checkbox("Show all columns (raw data)"):
            st.dataframe(df)
            raw_csv = df.to_csv(index=False)
            st.download_button(
                "Download Raw Data (All Columns)",
                data=raw_csv,
                file_name="raw_transactions_data.csv",
                mime="text/csv"
            )
    
    return df
import pandas as pd
import numpy as np
import re
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def categorize_transaction(description):
    """
    Categorize a transaction based on its description.
    
    Args:
        description (str): Transaction description
        
    Returns:
        str: Category of the transaction
    """
    description = description.lower()
    
    # Define category patterns
    categories = {
        "Food & Dining": ["coffee", "restaurant", "pizza", "cafe", "food", "dining", "meal", "breakfast", "lunch", "dinner", "snack", "bakery"],
        "Shopping": ["amazon", "shop", "store", "mall", "retail", "purchase", "buy", "market", "supermarket", "mart", "accessorie", "mobile"],
        "Transportation": ["uber", "cab", "taxi", "auto", "bus", "train", "metro", "fuel", "petrol", "gas", "diesel", "parking", "toll", "car", "bike", "scooter", "vehicle", "cargo", "transport"],
        "Entertainment": ["movie", "concert", "show", "ticket", "event", "game", "play", "sport", "netflix", "amazon prime", "hotstar", "subscription", "streaming"],
        "Bills & Utilities": ["bill", "electricity", "water", "internet", "gas", "utility", "phone", "mobile", "broadband", "cable", "dth", "recharge"],
        "Health & Wellness": ["medical", "doctor", "hospital", "pharmacy", "medicine", "health", "clinic", "dental", "fitness", "gym", "yoga", "spa", "wellness"],
        "Education": ["college", "school", "university", "course", "class", "tuition", "book", "stationery", "education", "learning", "training", "workshop"],
        "Travel": ["hotel", "resort", "airbnb", "flight", "air", "ticket", "vacation", "trip", "booking", "travel", "tour", "holiday"],
        "Investments": ["invest", "mutual fund", "stock", "share", "bond", "deposit", "fd", "dividend", "interest", "return", "capital", "profit"],
        "Salary & Income": ["salary", "income", "payroll", "commission", "bonus", "incentive", "allowance", "earning"],
        "Transfer": ["transfer", "send", "receive", "payment", "pay", "upi", "neft", "rtgs", "imps", "bank", "account", "wallet"],
        "Cash Withdrawal": ["atm", "withdraw", "cash"],
    }
    
    # Check for category matches
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in description:
                return category
    
    # Default category
    return "Miscellaneous"

def analyze_transactions(df):
    """
    Analyze transactions to provide insights and suggestions.
    
    Args:
        df (pandas.DataFrame): DataFrame containing transaction data
        
    Returns:
        dict: Analysis results and suggestions
    """
    if df.empty:
        return {
            "status": "error",
            "message": "No transactions to analyze"
        }
    
    try:
        # Ensure required columns exist
        required_cols = ["Date", "Transaction Details", "Debit", "Credit"]
        for col in required_cols:
            if col not in df.columns:
                return {
                    "status": "error",
                    "message": f"Missing required column: {col}"
                }
        
        # Create a working copy
        analysis_df = df.copy()
        
        # Add category column
        analysis_df["Category"] = analysis_df["Transaction Details"].apply(categorize_transaction)
        
        # Prepare data for analysis
        analysis_df["Amount"] = analysis_df["Debit"].fillna(0) - analysis_df["Credit"].fillna(0)
        analysis_df["Type"] = analysis_df["Amount"].apply(lambda x: "Expense" if x > 0 else "Income")
        analysis_df["Amount_Abs"] = analysis_df["Amount"].abs()
        
        # Basic statistics
        total_expenses = analysis_df[analysis_df["Type"] == "Expense"]["Amount_Abs"].sum()
        total_income = analysis_df[analysis_df["Type"] == "Income"]["Amount_Abs"].sum()
        net_cashflow = total_income - total_expenses
        
        # Category analysis
        category_expenses = analysis_df[analysis_df["Type"] == "Expense"].groupby("Category")["Amount_Abs"].sum().reset_index()
        category_expenses = category_expenses.sort_values("Amount_Abs", ascending=False)
        
        # Date analysis
        analysis_df["Week"] = analysis_df["Date"].dt.isocalendar().week
        weekly_expenses = analysis_df[analysis_df["Type"] == "Expense"].groupby("Week")["Amount_Abs"].sum().reset_index()
        
        # Top expense categories
        top_categories = category_expenses.head(3)["Category"].tolist()
        top_expenses = category_expenses.head(3)["Amount_Abs"].tolist()
        
        # Check for high-frequency small transactions
        small_transactions = analysis_df[
            (analysis_df["Type"] == "Expense") & 
            (analysis_df["Amount_Abs"] < analysis_df[analysis_df["Type"] == "Expense"]["Amount_Abs"].mean() * 0.5)
        ]
        high_freq_small = len(small_transactions) > len(analysis_df) * 0.3
        
        # Check for large one-time expenses
        large_transactions = analysis_df[
            (analysis_df["Type"] == "Expense") & 
            (analysis_df["Amount_Abs"] > analysis_df[analysis_df["Type"] == "Expense"]["Amount_Abs"].mean() * 2)
        ]
        has_large_expenses = len(large_transactions) > 0
        
        # Check spending consistency
        if len(weekly_expenses) > 1:
            spending_std = weekly_expenses["Amount_Abs"].std()
            spending_mean = weekly_expenses["Amount_Abs"].mean()
            spending_volatility = spending_std / spending_mean if spending_mean > 0 else 0
            high_volatility = spending_volatility > 0.5
        else:
            high_volatility = False
        
        # Check income vs. expenses
        expense_to_income_ratio = total_expenses / total_income if total_income > 0 else float('inf')
        high_expense_ratio = expense_to_income_ratio > 0.7
        
        # Generate insights and suggestions
        insights = []
        suggestions = []
        
        # Basic financial health
        if high_expense_ratio:
            insights.append(f"Your expenses represent {expense_to_income_ratio:.1%} of your income.")
            suggestions.append("Consider reducing expenses to maintain a healthier financial buffer.")
        else:
            insights.append(f"Your expense-to-income ratio is {expense_to_income_ratio:.1%}, which is healthy.")
            suggestions.append("Keep maintaining this healthy balance between income and expenses.")
        
        # Top spending categories
        if top_categories:
            insights.append(f"Your top spending categories are {', '.join(top_categories)}.")
            suggestions.append(f"Review your spending in {top_categories[0]} to identify potential savings.")
        
        # Spending patterns
        if high_volatility:
            insights.append("Your weekly spending shows high variation.")
            suggestions.append("Consider creating a budget to maintain more consistent spending patterns.")
        
        # Small frequent transactions
        if high_freq_small:
            insights.append("You have many small transactions that add up significantly.")
            suggestions.append("Track small daily expenses like coffee or snacks that can accumulate over time.")
        
        # Large expenses
        if has_large_expenses:
            large_categories = large_transactions["Category"].unique().tolist()
            insights.append(f"You have significant one-time expenses in {', '.join(large_categories)}.")
            suggestions.append("Plan ahead for large expenses to minimize their impact on your finances.")
        
        # Check for recurring subscriptions
        potential_subscriptions = analysis_df[
            (analysis_df["Type"] == "Expense") &
            (analysis_df["Amount_Abs"] < analysis_df[analysis_df["Type"] == "Expense"]["Amount_Abs"].mean()) &
            (analysis_df["Category"].isin(["Entertainment", "Bills & Utilities"]))
        ]
        
        if len(potential_subscriptions) > 0:
            insights.append("You might have recurring subscription payments.")
            suggestions.append("Review your subscriptions to eliminate services you don't use regularly.")
        
        # Overall financial advice
        if net_cashflow < 0:
            insights.append("Your expenses exceed your income during this period.")
            suggestions.append("Create a strict budget to bring your finances back into balance.")
        elif net_cashflow > 0:
            savings_rate = net_cashflow / total_income if total_income > 0 else 0
            insights.append(f"You're saving {savings_rate:.1%} of your income.")
            if savings_rate < 0.2:
                suggestions.append("Try to increase your savings rate to at least 20% of income.")
            else:
                suggestions.append("Consider investing your savings for long-term growth.")
        
        return {
            "status": "success",
            "statistics": {
                "total_expenses": total_expenses,
                "total_income": total_income,
                "net_cashflow": net_cashflow,
                "expense_ratio": expense_to_income_ratio,
                "transaction_count": len(analysis_df),
                "categories": category_expenses.to_dict(orient="records"),
            },
            "insights": insights,
            "suggestions": suggestions,
            "category_data": analysis_df
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error during analysis: {str(e)}"
        }

def display_expense_analysis(df):
    """
    Display expense analysis and suggestions in Streamlit.
    
    Args:
        df (pandas.DataFrame): DataFrame containing transaction data
    """
    st.markdown("## 💰 Expense Analysis & Financial Insights")
    
    # Run analysis
    analysis_results = analyze_transactions(df)
    
    if analysis_results["status"] == "error":
        st.error(analysis_results["message"])
        return
    
    # Get analysis data
    stats = analysis_results["statistics"]
    insights = analysis_results["insights"]
    suggestions = analysis_results["suggestions"]
    category_df = analysis_results["category_data"]
    
    # Display financial overview
    st.markdown("### Financial Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Income", f"₹ {stats['total_income']:,.2f}")
    
    with col2:
        st.metric("Total Expenses", f"₹ {stats['total_expenses']:,.2f}")
    
    with col3:
        net_value = stats['net_cashflow']
        delta_color = "normal" if net_value >= 0 else "inverse"
        st.metric("Net Cash Flow", f"₹ {net_value:,.2f}", delta=f"{'Positive' if net_value >= 0 else 'Negative'}", delta_color=delta_color)
    
    # Display expense breakdown
    st.markdown("### Expense Breakdown by Category")
    
    # Filter only expenses
    expense_df = category_df[category_df["Type"] == "Expense"]
    
    if not expense_df.empty:
        # Create category summary
        category_summary = expense_df.groupby("Category")["Amount_Abs"].agg(["sum", "count"]).reset_index()
        category_summary.columns = ["Category", "Total Amount", "Transaction Count"]
        category_summary = category_summary.sort_values("Total Amount", ascending=False)
        
        # Create pie chart for category distribution
        fig = px.pie(
            category_summary, 
            values="Total Amount", 
            names="Category",
            title="Expense Distribution by Category",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            margin=dict(t=50, b=100, l=10, r=10)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display category details
        st.markdown("#### Category Details")
        formatted_category = category_summary.copy()
        formatted_category["Total Amount"] = formatted_category["Total Amount"].apply(lambda x: f"₹ {x:,.2f}")
        st.dataframe(formatted_category, use_container_width=True, hide_index=True)
        
        # Display time-based analysis if enough data
        if len(expense_df["Date"].dt.date.unique()) > 3:
            st.markdown("### Spending Trends")
            
            # Weekly trend
            weekly_data = expense_df.groupby(expense_df["Date"].dt.isocalendar().week)["Amount_Abs"].sum().reset_index()
            weekly_data.columns = ["Week", "Amount"]
            
            # Create line chart
            fig_trend = px.line(
                weekly_data, 
                x="Week", 
                y="Amount",
                title="Weekly Spending Trend",
                markers=True
            )
            fig_trend.update_layout(
                xaxis_title="Week",
                yaxis_title="Amount (₹)",
                yaxis=dict(tickformat=",.0f")
            )
            st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No expense data available for analysis.")
    
    # Display insights and suggestions
    st.markdown("### 💡 Financial Insights")
    for i, insight in enumerate(insights):
        st.write(f"{i+1}. {insight}")
    
    st.markdown("### 🎯 Suggestions for Improvement")
    for i, suggestion in enumerate(suggestions):
        st.write(f"{i+1}. {suggestion}")
    
    # Final advice based on analysis
    st.markdown("### 📋 Personalized Action Plan")
    
    if stats['expense_ratio'] > 0.9:
        st.error("⚠️ Your expenses are too high relative to your income. Focus on reducing expenses immediately.")
    elif stats['expense_ratio'] > 0.7:
        st.warning("⚠️ Your expense-to-income ratio is concerning. Look for ways to reduce spending or increase income.")
    else:
        st.success("✅ Your overall financial health looks good. Focus on optimizing and growing your savings.")
    
    # Top 3 action items
    st.markdown("#### Top 3 Action Items:")
    top_actions = suggestions[:3] if len(suggestions) >= 3 else suggestions
    for i, action in enumerate(top_actions):
        st.write(f"**{i+1}.** {action}")
    
    # Add information about investing if user has positive net cash flow
    if stats['net_cashflow'] > 0:
        st.markdown("#### Investment Opportunities:")
        st.info("""
        💰 With a positive cash flow, consider these investment options:
        
        1. **Emergency Fund**: First, build a 3-6 month emergency fund
        2. **Retirement Accounts**: Contribute to retirement plans
        3. **Index Funds**: Low-cost way to invest in the market
        4. **Fixed Deposits**: For short-term, low-risk savings
        """)
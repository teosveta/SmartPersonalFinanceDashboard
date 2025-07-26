"""
Ultra-simple dashboard that works with minimal dependencies
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

st.set_page_config(page_title="Quick Finance Dashboard", page_icon="💰")


def generate_sample_data():
    """Generate sample data if database isn't available"""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')

    categories = ['Food & Dining', 'Transportation', 'Shopping', 'Entertainment', 'Bills & Utilities']
    merchants = ['Store A', 'Store B', 'Restaurant C', 'Gas Station D', 'Online Shop E']

    data = []
    for date in dates:
        # Generate 0-5 transactions per day
        num_transactions = random.randint(0, 5)
        for _ in range(num_transactions):
            data.append({
                'date': date,
                'amount': random.uniform(5, 200),
                'category': random.choice(categories),
                'merchant': random.choice(merchants),
                'description': 'Sample transaction'
            })

    return pd.DataFrame(data)


def quick_main():
    st.title("💰 Quick Finance Dashboard")

    st.info("🚀 This is a simplified version. For full features, ensure all files are properly set up.")

    # Generate or load data
    if st.button("🎲 Generate Sample Data"):
        df = generate_sample_data()
        st.session_state['quick_data'] = df
        st.success("Sample data generated!")

    # Load data if available
    if 'quick_data' in st.session_state:
        df = st.session_state['quick_data']

        # Basic metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Spending", f"${df['amount'].sum():,.2f}")

        with col2:
            st.metric("Transactions", f"{len(df):,}")

        with col3:
            st.metric("Avg Transaction", f"${df['amount'].mean():.2f}")

        with col4:
            st.metric("Categories", f"{df['category'].nunique()}")

        # Simple charts
        import plotly.express as px

        # Monthly spending
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
        monthly = df.groupby('month')['amount'].sum().reset_index()
        monthly['month'] = monthly['month'].astype(str)

        fig1 = px.line(monthly, x='month', y='amount', title='Monthly Spending')
        st.plotly_chart(fig1, use_container_width=True)

        # Category breakdown
        category_data = df.groupby('category')['amount'].sum().reset_index()
        fig2 = px.pie(category_data, values='amount', names='category', title='Spending by Category')
        st.plotly_chart(fig2, use_container_width=True)

        # Recent transactions
        st.subheader("Recent Transactions")
        recent = df.nlargest(10, 'amount')[['date', 'amount', 'category', 'merchant']]
        st.dataframe(recent, use_container_width=True)

    else:
        st.info("👆 Click 'Generate Sample Data' to start exploring!")


# Run quick version if this file is executed directly
if __name__ == "__main__":
    quick_main()
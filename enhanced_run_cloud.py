import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Page configuration
st.set_page_config(
    page_title="Smart Personal Finance Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Initialize session state
def initialize_session_state():
    defaults = {
        'data_loaded': False,
        'sample_data': None,
        'user_preferences': {
            'theme': 'light',
            'months_generated': 12
        }
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# Simple data generator for cloud deployment
def generate_cloud_sample_data(months=12, transactions_per_day=3):
    """Generate sample data without complex dependencies"""
    import random
    from datetime import datetime, timedelta

    # Set seed for reproducible data
    random.seed(42)
    np.random.seed(42)

    categories = {
        'Food & Dining': {'weight': 0.3, 'range': (5, 150)},
        'Transportation': {'weight': 0.25, 'range': (10, 200)},
        'Shopping': {'weight': 0.2, 'range': (15, 500)},
        'Entertainment': {'weight': 0.15, 'range': (8, 300)},
        'Bills & Utilities': {'weight': 0.1, 'range': (50, 300)}
    }

    merchants = {
        'Food & Dining': ['Starbucks', 'McDonald\'s', 'Whole Foods', 'Local Restaurant'],
        'Transportation': ['Shell', 'Uber', 'Metro Transit', 'Parking Garage'],
        'Shopping': ['Amazon', 'Target', 'Best Buy', 'Local Store'],
        'Entertainment': ['Netflix', 'Movie Theater', 'Spotify', 'Concert Hall'],
        'Bills & Utilities': ['Electric Company', 'Internet Provider', 'Insurance Co', 'Phone Company']
    }

    payment_methods = ['Credit Card', 'Debit Card', 'Cash', 'Digital Wallet']

    # Generate transactions
    transactions = []
    start_date = datetime.now() - timedelta(days=months * 30)
    end_date = datetime.now()

    current_date = start_date
    while current_date <= end_date:
        # More transactions on weekends
        day_multiplier = 1.3 if current_date.weekday() >= 5 else 1.0
        daily_count = max(1, int(np.random.poisson(transactions_per_day * day_multiplier)))

        for _ in range(daily_count):
            # Select category based on weights
            category = np.random.choice(
                list(categories.keys()),
                p=[categories[cat]['weight'] for cat in categories.keys()]
            )

            # Generate amount with seasonal variation
            min_amt, max_amt = categories[category]['range']
            seasonal_mult = 1.4 if current_date.month == 12 else 0.8 if current_date.month == 1 else 1.0
            amount = round(random.uniform(min_amt, max_amt) * seasonal_mult, 2)

            # Select merchant and other details
            merchant = random.choice(merchants[category])
            payment_method = random.choice(payment_methods)

            transactions.append({
                'date': current_date.date(),
                'amount': amount,
                'category': category,
                'merchant': merchant,
                'description': f'Transaction at {merchant}',
                'payment_method': payment_method
            })

        current_date += timedelta(days=1)

    return pd.DataFrame(transactions)


# Main dashboard function
def run_cloud_dashboard():
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h1 style='color: #2E86AB; font-size: 3rem; margin-bottom: 0;'>
            💰 Smart Personal Finance Dashboard
        </h1>
        <p style='color: #666; font-size: 1.2rem;'>
            AI-Powered Financial Analytics Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown("## 🎛️ Dashboard Controls")

    # Data generation section
    st.sidebar.markdown("### 💾 Data Management")

    if st.sidebar.button("🎲 Generate Sample Data", type="primary"):
        months = st.sidebar.slider("Months of data", 3, 24, 12)
        transactions_per_day = st.sidebar.slider("Avg transactions/day", 1, 8, 3)

        with st.spinner("Generating sample financial data..."):
            sample_data = generate_cloud_sample_data(months, transactions_per_day)
            st.session_state.sample_data = sample_data
            st.session_state.data_loaded = True

        st.sidebar.success(f"✅ Generated {len(sample_data):,} transactions!")

    # Check if data exists
    if not st.session_state.data_loaded or st.session_state.sample_data is None:
        # No data state
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 10px; margin: 2rem 0;'>
                <h2 style='color: #2E86AB;'>🚀 Welcome to Your Finance Dashboard!</h2>
                <p style='font-size: 1.1rem; margin: 1.5rem 0;'>
                    Let's start by generating some sample financial data to explore the dashboard features.
                </p>
                <p style='color: #666;'>
                    👈 Click <strong>"Generate Sample Data"</strong> in the sidebar to begin
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Feature preview
            st.markdown("### 🌟 Dashboard Features")

            feature_col1, feature_col2 = st.columns(2)

            with feature_col1:
                st.markdown("""
                **📊 Analytics & Insights**
                - Interactive spending charts
                - Category breakdowns
                - Monthly trend analysis
                - Financial health scoring
                """)

            with feature_col2:
                st.markdown("""
                **🎯 Smart Features**
                - Spending pattern detection
                - Budget vs actual tracking
                - Data export capabilities
                - Responsive filtering
                """)

        return

    # Data is available - show dashboard
    df = st.session_state.sample_data

    # Sidebar filters
    st.sidebar.markdown("### 🔍 Filters")

    # Date filter
    if 'date' in df.columns:
        min_date = pd.to_datetime(df['date']).min().date()
        max_date = pd.to_datetime(df['date']).max().date()

        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        # Apply date filter
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[
                (pd.to_datetime(df['date']).dt.date >= start_date) &
                (pd.to_datetime(df['date']).dt.date <= end_date)
                ]

    # Category filter
    if 'category' in df.columns:
        categories = ['All'] + list(df['category'].unique())
        selected_categories = st.sidebar.multiselect(
            "Categories",
            categories,
            default=['All']
        )

        if selected_categories and 'All' not in selected_categories:
            df = df[df['category'].isin(selected_categories)]

    # Quick stats sidebar
    st.sidebar.markdown("### 📊 Quick Stats")
    st.sidebar.metric("Total Spending", f"${df['amount'].sum():,.0f}")
    st.sidebar.metric("Transactions", f"{len(df):,}")
    st.sidebar.metric("Avg Transaction", f"${df['amount'].mean():.0f}")

    # Main dashboard content
    if df.empty:
        st.warning("No data matches your filters. Try adjusting the filter settings.")
        return

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_spending = df['amount'].sum()
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
            <h3 style='margin: 0; font-size: 1.1rem;'>💰 Total Spending</h3>
            <h2 style='margin: 0.5rem 0 0 0; font-size: 2rem;'>${total_spending:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        transaction_count = len(df)
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
            <h3 style='margin: 0; font-size: 1.1rem;'>🔢 Transactions</h3>
            <h2 style='margin: 0.5rem 0 0 0; font-size: 2rem;'>{transaction_count:,}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_transaction = df['amount'].mean()
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
            <h3 style='margin: 0; font-size: 1.1rem;'>📊 Average</h3>
            <h2 style='margin: 0.5rem 0 0 0; font-size: 2rem;'>${avg_transaction:.0f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        top_category = df.groupby('category')['amount'].sum().idxmax()
        top_amount = df.groupby('category')['amount'].sum().max()
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
            <h3 style='margin: 0; font-size: 1.1rem;'>🏆 Top Category</h3>
            <h4 style='margin: 0.2rem 0; font-size: 1rem;'>{top_category}</h4>
            <h3 style='margin: 0; font-size: 1.5rem;'>${top_amount:,.0f}</h3>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Charts section
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Monthly spending trend
        if 'date' in df.columns:
            import plotly.express as px

            df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
            monthly_data = df.groupby('month')['amount'].sum().reset_index()
            monthly_data['month'] = monthly_data['month'].astype(str)

            fig1 = px.line(
                monthly_data,
                x='month',
                y='amount',
                title='📈 Monthly Spending Trend',
                labels={'amount': 'Amount ($)', 'month': 'Month'}
            )
            fig1.update_layout(
                height=400,
                title_font_size=16,
                showlegend=False
            )
            st.plotly_chart(fig1, use_container_width=True)

    with chart_col2:
        # Category breakdown
        if 'category' in df.columns:
            category_data = df.groupby('category')['amount'].sum().reset_index()

            fig2 = px.pie(
                category_data,
                values='amount',
                names='category',
                title='🥧 Spending by Category'
            )
            fig2.update_layout(
                height=400,
                title_font_size=16
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Detailed analysis tabs
    tab1, tab2, tab3 = st.tabs(["📊 Category Analysis", "📋 Transaction Details", "📤 Export Data"])

    with tab1:
        st.subheader("Category Spending Analysis")

        if 'category' in df.columns:
            category_summary = df.groupby('category').agg({
                'amount': ['sum', 'count', 'mean', 'std']
            }).round(2)
            category_summary.columns = ['Total ($)', 'Count', 'Average ($)', 'Std Dev ($)']
            category_summary = category_summary.sort_values('Total ($)', ascending=False)

            # Add percentage column
            category_summary['Percentage (%)'] = (
                    category_summary['Total ($)'] / category_summary['Total ($)'].sum() * 100
            ).round(1)

            st.dataframe(category_summary, use_container_width=True)

            # Top merchants by category
            st.subheader("Top Merchants by Category")
            selected_cat = st.selectbox("Select Category", df['category'].unique())

            cat_merchants = df[df['category'] == selected_cat].groupby('merchant')['amount'].agg(
                ['sum', 'count']).sort_values('sum', ascending=False).head(10)
            cat_merchants.columns = ['Total Spent ($)', 'Transaction Count']

            st.dataframe(cat_merchants, use_container_width=True)

    with tab2:
        st.subheader("Recent Transactions")

        # Show recent high-value transactions
        recent_transactions = df.nlargest(20, 'amount')[
            ['date', 'amount', 'category', 'merchant', 'description', 'payment_method']
        ].copy()

        # Format the amount column
        recent_transactions['amount'] = recent_transactions['amount'].apply(lambda x: f"${x:.2f}")

        st.dataframe(recent_transactions, use_container_width=True)

        # Transaction summary
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Date Range", f"{df['date'].min()} to {df['date'].max()}")

        with col2:
            st.metric("Unique Merchants", f"{df['merchant'].nunique()}")

        with col3:
            st.metric("Payment Methods", f"{df['payment_method'].nunique()}")

    with tab3:
        st.subheader("Export Your Data")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Export Options:**")
            st.markdown("• Full transaction data as CSV")
            st.markdown("• Category summary report")
            st.markdown("• Monthly spending analysis")

            # CSV export
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📄 Download Full Data (CSV)",
                data=csv_data,
                file_name=f"financial_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                type="primary"
            )

        with col2:
            st.markdown("**Data Summary:**")
            st.write(f"• **Total Records**: {len(df):,}")
            st.write(
                f"• **Date Range**: {(pd.to_datetime(df['date']).max() - pd.to_datetime(df['date']).min()).days} days")
            st.write(f"• **Categories**: {df['category'].nunique()}")
            st.write(f"• **Merchants**: {df['merchant'].nunique()}")
            st.write(f"• **Total Amount**: ${df['amount'].sum():,.2f}")


# Run the dashboard
run_cloud_dashboard()
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)


# Initialize session state BEFORE any other streamlit operations
def initialize_session_state():
    """Initialize all session state variables - must be called early"""
    default_states = {
        'data_loaded': False,
        'models_trained': False,
        'advanced_models_trained': False,
        'last_data_refresh': None,
        'user_preferences': {
            'theme': 'light',
            'auto_refresh': False,
            'refresh_interval': 300  # 5 minutes
        },
        'budget_targets': {
            'Food & Dining': 800,
            'Transportation': 400,
            'Shopping': 300,
            'Entertainment': 200,
            'Bills & Utilities': 600
        },
        'savings_goals': {
            'emergency_fund': 5000,
            'vacation': 2000,
            'monthly_target': 500
        },
        'alert_preferences': {
            'budget_threshold': 80,
            'anomaly_sensitivity': 'medium',
            'email_notifications': False
        },
        'dashboard_theme': 'light'
    }

    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value


# Initialize session state immediately
initialize_session_state()

# Import with error handling
try:
    from src.utils.database import DatabaseManager
    from src.etl.data_loader import DataLoader
    from src.data_generation.transaction_generator import TransactionGenerator

    # Try to import enhanced features, but don't fail if they're missing
    try:
        from src.models.advanced_predictor import EnsembleExpensePredictor

        ADVANCED_FEATURES_AVAILABLE = True
    except ImportError:
        ADVANCED_FEATURES_AVAILABLE = False
        st.warning("Advanced ML features not available - some files missing")
except ImportError as e:
    st.error(f"Critical import error: {e}")
    st.error("Please ensure core files are properly installed.")
    st.stop()

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 10px 10px 0;
    }
    .warning-box {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 10px 10px 0;
    }
</style>
""", unsafe_allow_html=True)


class EnhancedFinanceDashboard:
    def __init__(self):
        try:
            self.db_manager = DatabaseManager()
            self.data_loader = DataLoader()

            # Initialize enhanced features if available
            if ADVANCED_FEATURES_AVAILABLE:
                self.ensemble_predictor = EnsembleExpensePredictor()

        except Exception as e:
            st.error(f"Error initializing dashboard components: {e}")
            st.stop()

    def load_data_safely(self):
        """Load data with comprehensive error handling"""
        try:
            self.df = self.data_loader.load_transactions()

            if not self.df.empty:
                st.session_state.data_loaded = True
                st.session_state.df = self.df
                logger.info(f"Loaded {len(self.df)} transactions")
                return True
            else:
                # No data found - show helpful message
                st.session_state.data_loaded = False
                return False

        except Exception as e:
            st.error(f"Error loading data: {e}")
            logger.error(f"Data loading error: {e}")
            return False

    def create_data_generation_interface(self):
        """Create interface for generating sample data"""
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("### 🎲 No Data Found")
        st.markdown("It looks like your database is empty. Let's generate some sample data to get started!")
        st.markdown("</div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.markdown("#### Generate Sample Financial Data")

            months_back = st.slider(
                "Months of data to generate:",
                min_value=3,
                max_value=24,
                value=12,
                help="More months = more data for better ML training"
            )

            transactions_per_day = st.slider(
                "Average transactions per day:",
                min_value=1,
                max_value=10,
                value=3,
                help="Higher values create more detailed spending patterns"
            )

            if st.button("🚀 Generate Sample Data", type="primary"):
                with st.spinner(f"Generating {months_back} months of sample transactions..."):
                    try:
                        generator = TransactionGenerator()
                        transactions = generator.generate_transactions(
                            start_date=datetime.now() - timedelta(days=months_back * 30),
                            end_date=datetime.now(),
                            avg_transactions_per_day=transactions_per_day
                        )
                        generator.save_to_database(transactions, self.db_manager)

                        st.success(f"✅ Generated {len(transactions):,} transactions!")
                        st.session_state.data_loaded = False  # Force reload
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error generating data: {e}")
                        logger.error(f"Data generation error: {e}")

            st.markdown("---")
            st.markdown("**What this creates:**")
            st.markdown("• Realistic spending patterns across 5 categories")
            st.markdown("• Seasonal variations (higher December spending, etc.)")
            st.markdown("• Weekend vs weekday differences")
            st.markdown("• Various merchants and payment methods")

    def create_simple_sidebar(self):
        """Create simplified sidebar when data is not available"""
        st.sidebar.markdown("<h1 style='color: #2E86AB;'>🎛️ Dashboard Controls</h1>", unsafe_allow_html=True)

        st.sidebar.markdown("### 💾 Data Status")
        if hasattr(self, 'df') and not self.df.empty:
            st.sidebar.success("✅ Data loaded")
            st.sidebar.metric("Transactions", f"{len(self.df):,}")
        else:
            st.sidebar.warning("⚠️ No data available")

        st.sidebar.markdown("### 🔄 Actions")
        if st.sidebar.button("🎲 Generate Data"):
            st.session_state.show_generation = True
            st.rerun()

        if st.sidebar.button("🔄 Refresh"):
            st.session_state.data_loaded = False
            st.rerun()

    def create_enhanced_sidebar(self):
        """Create enhanced sidebar when data is available"""
        st.sidebar.markdown("<h1 style='color: #2E86AB;'>🎛️ Dashboard Controls</h1>", unsafe_allow_html=True)

        # Quick stats
        if hasattr(self, 'df') and not self.df.empty:
            st.sidebar.markdown("### 📊 Quick Stats")
            total_spending = self.df['amount'].sum()
            transaction_count = len(self.df)
            avg_transaction = self.df['amount'].mean()

            st.sidebar.metric("Total Spending", f"${total_spending:,.0f}")
            st.sidebar.metric("Transactions", f"{transaction_count:,}")
            st.sidebar.metric("Avg Transaction", f"${avg_transaction:.0f}")
            st.sidebar.markdown("---")

        # Data management
        st.sidebar.markdown("### 💾 Data Management")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🔄 Refresh", key="refresh_data"):
                st.session_state.data_loaded = False
                st.rerun()

        with col2:
            if st.button("➕ Add Data", key="add_data"):
                st.session_state.show_generation = True
                st.rerun()

        # Filters (only if data exists)
        if hasattr(self, 'df') and not self.df.empty and 'date' in self.df.columns:
            st.sidebar.markdown("### 🔍 Filters")

            try:
                min_date = pd.to_datetime(self.df['date']).min().date()
                max_date = pd.to_datetime(self.df['date']).max().date()

                date_range = st.sidebar.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )

                # Category filter
                categories = ['All'] + list(self.df['category'].unique())
                selected_categories = st.sidebar.multiselect(
                    "Categories",
                    categories,
                    default=['All']
                )

                return {
                    'date_range': date_range,
                    'selected_categories': selected_categories
                }
            except Exception as e:
                st.sidebar.error(f"Filter error: {e}")

        return None

    def apply_filters_safely(self, filters):
        """Apply filters with error handling"""
        if not filters or not hasattr(self, 'df') or self.df.empty:
            return getattr(self, 'df', pd.DataFrame())

        filtered_df = self.df.copy()

        try:
            # Date filter
            if filters.get('date_range') and len(filters['date_range']) == 2:
                start_date, end_date = filters['date_range']
                if 'date' in filtered_df.columns:
                    filtered_df = filtered_df[
                        (pd.to_datetime(filtered_df['date']).dt.date >= start_date) &
                        (pd.to_datetime(filtered_df['date']).dt.date <= end_date)
                        ]

            # Category filter
            if (filters.get('selected_categories') and
                    'All' not in filters['selected_categories'] and
                    'category' in filtered_df.columns):
                filtered_df = filtered_df[filtered_df['category'].isin(filters['selected_categories'])]

        except Exception as e:
            st.error(f"Error applying filters: {e}")
            return self.df

        return filtered_df

    def create_overview_with_data(self, df):
        """Create overview when data is available"""
        st.markdown("<h2 class='main-header'>📊 Financial Dashboard</h2>", unsafe_allow_html=True)

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_spending = df['amount'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <h3>💰 Total Spending</h3>
                <h2>${total_spending:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            if 'date' in df.columns:
                date_range_days = (pd.to_datetime(df['date']).max() - pd.to_datetime(df['date']).min()).days
                avg_daily = total_spending / max(1, date_range_days)
            else:
                avg_daily = total_spending / len(df)

            st.markdown(f"""
            <div class="metric-card">
                <h3>📅 Daily Average</h3>
                <h2>${avg_daily:.0f}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            transaction_count = len(df)
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔢 Transactions</h3>
                <h2>{transaction_count:,}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            if 'category' in df.columns:
                top_category = df.groupby('category')['amount'].sum().idxmax()
                top_category_amount = df.groupby('category')['amount'].sum().max()
            else:
                top_category = "N/A"
                top_category_amount = 0

            st.markdown(f"""
            <div class="metric-card">
                <h3>🏆 Top Category</h3>
                <h4>{top_category}</h4>
                <h3>${top_category_amount:,.0f}</h3>
            </div>
            """, unsafe_allow_html=True)

        # Charts
        if 'date' in df.columns and 'category' in df.columns:
            import plotly.express as px

            col1, col2 = st.columns(2)

            with col1:
                # Monthly spending trend
                try:
                    df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
                    monthly_data = df.groupby('month')['amount'].sum().reset_index()
                    monthly_data['month'] = monthly_data['month'].astype(str)

                    fig1 = px.line(monthly_data, x='month', y='amount',
                                   title='Monthly Spending Trend',
                                   labels={'amount': 'Amount ($)', 'month': 'Month'})
                    fig1.update_layout(height=400)
                    st.plotly_chart(fig1, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating trend chart: {e}")

            with col2:
                # Category breakdown
                try:
                    category_data = df.groupby('category')['amount'].sum().reset_index()
                    fig2 = px.pie(category_data, values='amount', names='category',
                                  title='Spending by Category')
                    fig2.update_layout(height=400)
                    st.plotly_chart(fig2, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating category chart: {e}")

        # Recent transactions
        st.subheader("📋 Recent High-Value Transactions")
        try:
            display_columns = [col for col in ['date', 'amount', 'category', 'merchant', 'description']
                               if col in df.columns]
            recent_transactions = df.nlargest(10, 'amount')[display_columns]
            st.dataframe(recent_transactions, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying transactions: {e}")

    def run(self):
        """Main dashboard application with comprehensive error handling"""
        try:
            # Header
            st.markdown("<h1 class='main-header'>💰 Smart Personal Finance Dashboard</h1>", unsafe_allow_html=True)

            # Check if we should show data generation
            if st.session_state.get('show_generation', False):
                self.create_data_generation_interface()
                st.session_state.show_generation = False
                return

            # Try to load data
            data_loaded = self.load_data_safely()

            if not data_loaded:
                # No data available - show generation interface
                self.create_simple_sidebar()
                self.create_data_generation_interface()
                return

            # Data is available - show full dashboard
            filters = self.create_enhanced_sidebar()
            filtered_df = self.apply_filters_safely(filters)

            if filtered_df.empty:
                st.warning("No data matches the selected filters. Try adjusting your filter settings.")
                return

            # Create main dashboard
            self.create_overview_with_data(filtered_df)

            # Additional tabs for data exploration
            st.markdown("---")

            tab1, tab2 = st.tabs(["📊 Data Analysis", "📋 Export & Info"])

            with tab1:
                if 'category' in filtered_df.columns:
                    st.subheader("📈 Category Analysis")

                    # Category spending breakdown
                    category_summary = filtered_df.groupby('category').agg({
                        'amount': ['sum', 'count', 'mean']
                    }).round(2)
                    category_summary.columns = ['Total ($)', 'Transactions', 'Average ($)']
                    category_summary = category_summary.sort_values('Total ($)', ascending=False)

                    st.dataframe(category_summary, use_container_width=True)
                else:
                    st.info("Category analysis not available - category column missing")

            with tab2:
                st.subheader("📋 Data Information")

                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Total Rows:** {len(filtered_df):,}")
                    st.write(f"**Columns:** {', '.join(filtered_df.columns)}")
                    if 'date' in filtered_df.columns:
                        st.write(f"**Date Range:** {filtered_df['date'].min()} to {filtered_df['date'].max()}")
                    if 'category' in filtered_df.columns:
                        st.write(f"**Categories:** {filtered_df['category'].nunique()}")

                with col2:
                    if st.button("📄 Export to CSV"):
                        csv = filtered_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"financial_data_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )

                # Show data sample
                st.subheader("📋 Data Sample")
                st.dataframe(filtered_df.head(20), use_container_width=True)

        except Exception as e:
            st.error(f"Dashboard error: {e}")
            logger.error(f"Dashboard error: {e}", exc_info=True)

            # Show emergency recovery options
            st.markdown("### 🆘 Recovery Options")
            if st.button("🔄 Reset Dashboard"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
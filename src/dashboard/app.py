import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Import with error handling
try:
    from src.utils.database import DatabaseManager
    from src.etl.data_loader import DataLoader
    from src.models.expense_predictor import ExpensePredictor
    from src.models.category_classifier import CategoryClassifier
    from src.models.anomaly_detector import AnomalyDetector
    from src.analytics.insights_generator import InsightsGenerator
    from src.dashboard.components.charts import ChartGenerator
    from src.data_generation.transaction_generator import TransactionGenerator
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all source files are properly installed.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Smart Personal Finance Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


class FinanceDashboard:
    def __init__(self):
        try:
            self.db_manager = DatabaseManager()
            self.data_loader = DataLoader()
            self.expense_predictor = ExpensePredictor()
            self.category_classifier = CategoryClassifier()
            self.anomaly_detector = AnomalyDetector()
            self.insights_generator = InsightsGenerator()
            self.chart_generator = ChartGenerator()
        except Exception as e:
            st.error(f"Error initializing dashboard components: {e}")
            st.stop()

        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'models_trained' not in st.session_state:
            st.session_state.models_trained = False

    def load_data(self):
        """Load and cache transaction data"""
        if not st.session_state.data_loaded:
            with st.spinner("Loading transaction data..."):
                try:
                    self.df = self.data_loader.load_transactions()
                    if not self.df.empty:
                        st.session_state.data_loaded = True
                        st.session_state.df = self.df
                        logger.info(f"Loaded {len(self.df)} transactions")
                    else:
                        st.warning("No transaction data found. Please generate sample data first.")
                        return False
                except Exception as e:
                    st.error(f"Error loading data: {e}")
                    return False
        else:
            self.df = st.session_state.df
        return True

    def train_models(self):
        """Train ML models"""
        if not st.session_state.models_trained and hasattr(self, 'df') and not self.df.empty:
            with st.spinner("Training ML models..."):
                try:
                    # Train category classifier
                    self.category_classifier.train(self.df)

                    # Train expense predictor
                    self.expense_predictor.train_ml_model(self.df)

                    # Train anomaly detector
                    self.anomaly_detector.train_isolation_forest(self.df)

                    st.session_state.models_trained = True
                    logger.info("All models trained successfully")
                except Exception as e:
                    st.error(f"Error training models: {e}")

    def sidebar_controls(self):
        """Create sidebar controls"""
        st.sidebar.title("Dashboard Controls")

        # Data management
        st.sidebar.header("Data Management")

        if st.sidebar.button("Generate Sample Data"):
            with st.spinner("Generating sample transactions..."):
                try:
                    generator = TransactionGenerator()
                    generator.generate_and_save(months_back=12, db_manager=self.db_manager)
                    st.session_state.data_loaded = False  # Force reload
                    st.sidebar.success("Sample data generated!")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Error generating data: {e}")

        # Date range filter
        st.sidebar.header("Filters")

        if hasattr(self, 'df') and not self.df.empty:
            try:
                min_date = pd.to_datetime(self.df['date']).min().date()
                max_date = pd.to_datetime(self.df['date']).max().date()

                date_range = st.sidebar.date_input(
                    "Select Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )

                # Category filter
                categories = ['All'] + list(self.df['category'].unique())
                selected_categories = st.sidebar.multiselect(
                    "Select Categories",
                    categories,
                    default=['All']
                )

                return date_range, selected_categories
            except Exception as e:
                st.sidebar.error(f"Error setting up filters: {e}")

        return None, None

    def filter_data(self, date_range, selected_categories):
        """Filter data based on sidebar controls"""
        filtered_df = self.df.copy()

        try:
            if date_range and len(date_range) == 2:
                start_date, end_date = date_range
                filtered_df = filtered_df[
                    (pd.to_datetime(filtered_df['date']).dt.date >= start_date) &
                    (pd.to_datetime(filtered_df['date']).dt.date <= end_date)
                    ]

            if selected_categories and 'All' not in selected_categories:
                filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
        except Exception as e:
            st.error(f"Error filtering data: {e}")
            return self.df

        return filtered_df

    def overview_tab(self, df):
        """Create overview tab content"""
        st.header("📊 Financial Overview")

        try:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_spending = df['amount'].sum()
                st.metric("Total Spending", f"${total_spending:,.2f}")

            with col2:
                avg_transaction = df['amount'].mean()
                st.metric("Avg Transaction", f"${avg_transaction:.2f}")

            with col3:
                transaction_count = len(df)
                st.metric("Total Transactions", f"{transaction_count:,}")

            with col4:
                if not df.empty:
                    top_category = df.groupby('category')['amount'].sum().idxmax()
                    st.metric("Top Category", top_category)

            # Overview charts
            if not df.empty:
                overview_chart = self.chart_generator.create_spending_overview(df)
                st.plotly_chart(overview_chart, use_container_width=True)

            # Recent transactions
            st.subheader("Recent Transactions")
            if not df.empty:
                recent_transactions = df.nlargest(10, 'amount')[
                    ['date', 'amount', 'category', 'merchant', 'description']
                ]
                st.dataframe(recent_transactions, use_container_width=True)
            else:
                st.info("No transactions to display")

        except Exception as e:
            st.error(f"Error in overview tab: {e}")

    def predictions_tab(self, df):
        """Create predictions tab content"""
        st.header("🔮 Expense Predictions")

        try:
            col1, col2 = st.columns([2, 1])

            with col2:
                st.subheader("Prediction Settings")
                prediction_days = st.slider("Days to Predict", 7, 90, 30)
                selected_category = st.selectbox(
                    "Category (All for total)",
                    ['All'] + list(df['category'].unique()) if not df.empty else ['All']
                )

            with col1:
                if not df.empty and st.session_state.models_trained:
                    # Generate predictions
                    category = None if selected_category == 'All' else selected_category

                    predictions = self.expense_predictor.predict_future_expenses(
                        df, days_ahead=prediction_days, category=category
                    )

                    if not predictions.empty:
                        # Time series chart with predictions
                        ts_chart = self.chart_generator.create_time_series_chart(df, predictions)
                        st.plotly_chart(ts_chart, use_container_width=True)

                        # Prediction summary
                        total_predicted = predictions['predicted_amount'].sum()
                        avg_daily_predicted = predictions['predicted_amount'].mean()

                        st.subheader("Prediction Summary")
                        pred_col1, pred_col2 = st.columns(2)
                        with pred_col1:
                            st.metric("Total Predicted", f"${total_predicted:.2f}")
                        with pred_col2:
                            st.metric("Avg Daily", f"${avg_daily_predicted:.2f}")
                    else:
                        st.warning("Unable to generate predictions with current data.")
                else:
                    if df.empty:
                        st.warning("No data available for predictions.")
                    else:
                        st.warning("Models not trained yet. Please wait for model training to complete.")
        except Exception as e:
            st.error(f"Error in predictions tab: {e}")

    def analytics_tab(self, df):
        """Create analytics tab content"""
        st.header("📈 Advanced Analytics")

        try:
            if not df.empty:
                # Category analysis
                st.subheader("Category Analysis")
                category_chart = self.chart_generator.create_category_analysis(df)
                st.plotly_chart(category_chart, use_container_width=True)

                # Anomaly detection
                st.subheader("Anomaly Detection")

                if st.session_state.models_trained:
                    with st.spinner("Detecting anomalies..."):
                        df_with_anomalies = self.anomaly_detector.detect_all_anomalies(df)

                        anomaly_chart = self.chart_generator.create_anomaly_chart(df_with_anomalies)
                        st.plotly_chart(anomaly_chart, use_container_width=True)

                        # Anomaly summary
                        anomaly_summary = self.anomaly_detector.get_anomaly_summary(df_with_anomalies)

                        if 'total_anomalies' in anomaly_summary:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Anomalies Detected", anomaly_summary['total_anomalies'])
                            with col2:
                                st.metric("Anomaly Rate", f"{anomaly_summary['anomaly_rate']:.1%}")
                            with col3:
                                st.metric("Max Anomaly", f"${anomaly_summary['max_anomaly_amount']:.2f}")

                            # Recent anomalies
                            if anomaly_summary['recent_anomalies']:
                                st.subheader("Recent Anomalies")
                                anomaly_df = pd.DataFrame(anomaly_summary['recent_anomalies'])
                                st.dataframe(anomaly_df, use_container_width=True)
                else:
                    st.warning("Please wait for models to finish training.")
            else:
                st.info("No data available for analytics.")
        except Exception as e:
            st.error(f"Error in analytics tab: {e}")

    def insights_tab(self, df):
        """Create insights tab content"""
        st.header("💡 Financial Insights")

        try:
            if not df.empty:
                # Generate insights
                insights_data = self.insights_generator.generate_all_insights(df)

                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Insights", insights_data['total_insights'])
                with col2:
                    st.metric("High Priority", insights_data['high_priority_insights'])
                with col3:
                    spending_summary = insights_data['spending_summary']
                    st.metric("Avg Transaction", f"${spending_summary['avg_transaction']:.2f}")

                # Display insights
                st.subheader("Key Insights")

                for insight in insights_data['insights']:
                    priority_color = {
                        'high': 'red',
                        'medium': 'orange',
                        'low': 'blue'
                    }.get(insight['priority'], 'gray')

                    with st.expander(f"🔍 {insight['title']} ({insight['priority']} priority)"):
                        st.write(insight['description'])
                        if 'value' in insight:
                            st.write(f"**Value:** {insight['value']:.2f}")
            else:
                st.info("No data available for insights generation.")
        except Exception as e:
            st.error(f"Error in insights tab: {e}")

    def run(self):
        """Main dashboard application"""
        st.title("💰 Smart Personal Finance Dashboard")
        st.markdown("---")

        try:
            # Sidebar controls
            date_range, selected_categories = self.sidebar_controls()

            # Load data
            if not self.load_data():
                st.stop()

            # Filter data
            filtered_df = self.filter_data(date_range, selected_categories)

            if filtered_df.empty:
                st.warning("No data available for the selected filters.")
                st.stop()

            # Train models if needed
            self.train_models()

            # Create tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Predictions", "Analytics", "Insights"])

            with tab1:
                self.overview_tab(filtered_df)

            with tab2:
                self.predictions_tab(filtered_df)

            with tab3:
                self.analytics_tab(filtered_df)

            with tab4:
                self.insights_tab(filtered_df)

        except Exception as e:
            st.error(f"Error running dashboard: {e}")
            logger.error(f"Dashboard error: {e}", exc_info=True)
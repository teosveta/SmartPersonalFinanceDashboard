import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np


class InteractiveFeatures:
    """Advanced interactive features for the dashboard"""

    def __init__(self):
        self.session_state_keys = [
            'budget_targets', 'savings_goals', 'alert_preferences', 'dashboard_theme'
        ]
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize session state for interactive features"""
        for key in self.session_state_keys:
            if key not in st.session_state:
                st.session_state[key] = self.get_default_value(key)

    def get_default_value(self, key: str):
        """Get default values for session state"""
        defaults = {
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
        return defaults.get(key, {})

    def create_budget_planner(self) -> dict:
        """Create interactive budget planning interface"""
        st.subheader("💰 Budget Planner")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.write("**Set Monthly Budget Targets:**")

            budget_targets = st.session_state['budget_targets'].copy()

            for category in budget_targets.keys():
                budget_targets[category] = st.number_input(
                    f"{category}",
                    min_value=0.0,
                    value=float(budget_targets[category]),
                    step=50.0,
                    key=f"budget_{category}"
                )

            if st.button("Save Budget Targets"):
                st.session_state['budget_targets'] = budget_targets
                st.success("Budget targets saved!")

        with col2:
            total_budget = sum(budget_targets.values())
            st.metric("Total Monthly Budget", f"${total_budget:,.2f}")

            # Budget allocation pie chart
            fig = go.Figure(data=[go.Pie(
                labels=list(budget_targets.keys()),
                values=list(budget_targets.values()),
                hole=0.3
            )])
            fig.update_layout(title="Budget Allocation", height=300)
            st.plotly_chart(fig, use_container_width=True)

        return budget_targets

    def create_savings_goal_tracker(self) -> dict:
        """Create interactive savings goal tracker"""
        st.subheader("🎯 Savings Goals")

        savings_goals = st.session_state['savings_goals'].copy()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Emergency Fund**")
            current_emergency = st.number_input(
                "Current Amount",
                min_value=0.0,
                value=0.0,
                key="current_emergency"
            )
            target_emergency = st.number_input(
                "Target Amount",
                min_value=0.0,
                value=float(savings_goals['emergency_fund']),
                key="target_emergency"
            )

            progress_emergency = min(current_emergency / target_emergency, 1.0) if target_emergency > 0 else 0
            st.progress(progress_emergency)
            st.write(f"Progress: {progress_emergency:.1%}")

        with col2:
            st.write("**Vacation Fund**")
            current_vacation = st.number_input(
                "Current Amount",
                min_value=0.0,
                value=0.0,
                key="current_vacation"
            )
            target_vacation = st.number_input(
                "Target Amount",
                min_value=0.0,
                value=float(savings_goals['vacation']),
                key="target_vacation"
            )

            progress_vacation = min(current_vacation / target_vacation, 1.0) if target_vacation > 0 else 0
            st.progress(progress_vacation)
            st.write(f"Progress: {progress_vacation:.1%}")

        with col3:
            st.write("**Monthly Savings**")
            monthly_target = st.number_input(
                "Monthly Target",
                min_value=0.0,
                value=float(savings_goals['monthly_target']),
                key="monthly_target"
            )

            # Calculate time to reach goals
            if monthly_target > 0:
                months_to_emergency = max(0, (target_emergency - current_emergency) / monthly_target)
                months_to_vacation = max(0, (target_vacation - current_vacation) / monthly_target)

                st.metric("Months to Emergency Goal", f"{months_to_emergency:.1f}")
                st.metric("Months to Vacation Goal", f"{months_to_vacation:.1f}")

        return {
            'emergency_fund': target_emergency,
            'vacation': target_vacation,
            'monthly_target': monthly_target,
            'current_emergency': current_emergency,
            'current_vacation': current_vacation
        }

    def create_alert_preferences(self) -> dict:
        """Create alert preferences interface"""
        st.subheader("🔔 Alert Preferences")

        alert_prefs = st.session_state['alert_preferences'].copy()

        col1, col2 = st.columns(2)

        with col1:
            alert_prefs['budget_threshold'] = st.slider(
                "Budget Alert Threshold (%)",
                min_value=50,
                max_value=100,
                value=alert_prefs['budget_threshold'],
                step=5,
                help="Get alerted when spending reaches this % of budget"
            )

            alert_prefs['anomaly_sensitivity'] = st.selectbox(
                "Anomaly Detection Sensitivity",
                options=['low', 'medium', 'high'],
                index=['low', 'medium', 'high'].index(alert_prefs['anomaly_sensitivity'])
            )

        with col2:
            alert_prefs['email_notifications'] = st.checkbox(
                "Enable Email Notifications",
                value=alert_prefs['email_notifications']
            )

            if alert_prefs['email_notifications']:
                email = st.text_input("Email Address", placeholder="your.email@example.com")
                alert_prefs['email'] = email

        if st.button("Save Alert Preferences"):
            st.session_state['alert_preferences'] = alert_prefs
            st.success("Alert preferences saved!")

        return alert_prefs

    def create_scenario_simulator(self, df: pd.DataFrame) -> dict:
        """Create financial scenario simulator"""
        st.subheader("🔮 Scenario Simulator")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.write("**Simulate Changes:**")

            scenario_type = st.selectbox(
                "Scenario Type",
                ["Income Change", "New Recurring Expense", "One-time Purchase", "Savings Rate Change"]
            )

            if scenario_type == "Income Change":
                income_change = st.number_input(
                    "Monthly Income Change ($)",
                    value=0.0,
                    step=100.0
                )
                scenario_params = {'income_change': income_change}

            elif scenario_type == "New Recurring Expense":
                expense_amount = st.number_input(
                    "Monthly Expense ($)",
                    min_value=0.0,
                    value=100.0,
                    step=25.0
                )
                expense_category = st.selectbox(
                    "Category",
                    df['category'].unique() if not df.empty else ['Other']
                )
                scenario_params = {
                    'expense_amount': expense_amount,
                    'expense_category': expense_category
                }

            elif scenario_type == "One-time Purchase":
                purchase_amount = st.number_input(
                    "Purchase Amount ($)",
                    min_value=0.0,
                    value=500.0,
                    step=100.0
                )
                scenario_params = {'purchase_amount': purchase_amount}

            else:  # Savings Rate Change
                savings_rate_change = st.slider(
                    "Savings Rate Change (%)",
                    min_value=-20,
                    max_value=20,
                    value=0,
                    step=1
                )
                scenario_params = {'savings_rate_change': savings_rate_change}

            simulate_months = st.slider(
                "Simulation Period (months)",
                min_value=1,
                max_value=24,
                value=12
            )

        with col2:
            if st.button("Run Simulation"):
                simulation_results = self.run_scenario_simulation(
                    df, scenario_type, scenario_params, simulate_months
                )

                # Display results
                st.write("**Simulation Results:**")

                # Create comparison chart
                fig = go.Figure()

                months = list(range(1, simulate_months + 1))

                # Baseline scenario
                fig.add_trace(go.Scatter(
                    x=months,
                    y=simulation_results['baseline'],
                    mode='lines',
                    name='Current Trajectory',
                    line=dict(color='blue', width=2)
                ))

                # Modified scenario
                fig.add_trace(go.Scatter(
                    x=months,
                    y=simulation_results['scenario'],
                    mode='lines',
                    name=f'With {scenario_type}',
                    line=dict(color='red', width=2, dash='dash')
                ))

                fig.update_layout(
                    title=f"Financial Impact of {scenario_type}",
                    xaxis_title="Months",
                    yaxis_title="Net Position ($)",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

                # Summary metrics
                final_difference = simulation_results['scenario'][-1] - simulation_results['baseline'][-1]
                st.metric(
                    f"Impact after {simulate_months} months",
                    f"${final_difference:,.2f}",
                    delta=f"{'Better' if final_difference > 0 else 'Worse'} than baseline"
                )

        return scenario_params if 'scenario_params' in locals() else {}

    def run_scenario_simulation(self, df: pd.DataFrame, scenario_type: str,
                                params: dict, months: int) -> dict:
        """Run financial scenario simulation"""
        if df.empty:
            # Generate dummy baseline data
            baseline = [1000 * i for i in range(1, months + 1)]
            scenario = baseline.copy()
            return {'baseline': baseline, 'scenario': scenario}

        # Calculate baseline monthly spending
        monthly_spending = df.groupby(df['date'].dt.to_period('M'))['amount'].sum().mean()

        # Simulate baseline trajectory
        baseline = []
        net_position = 0

        for month in range(months):
            net_position += -monthly_spending  # Negative because it's spending
            baseline.append(net_position)

        # Simulate scenario
        scenario = []
        net_position = 0

        for month in range(months):
            monthly_impact = -monthly_spending

            if scenario_type == "Income Change":
                monthly_impact += params['income_change']
            elif scenario_type == "New Recurring Expense":
                monthly_impact -= params['expense_amount']
            elif scenario_type == "One-time Purchase" and month == 0:
                monthly_impact -= params['purchase_amount']
            elif scenario_type == "Savings Rate Change":
                savings_impact = monthly_spending * (params['savings_rate_change'] / 100)
                monthly_impact += savings_impact

            net_position += monthly_impact
            scenario.append(net_position)

        return {'baseline': baseline, 'scenario': scenario}

    def create_export_functionality(self, df: pd.DataFrame):
        """Create data export functionality"""
        st.subheader("📊 Export Data")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Export to CSV"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"financial_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

        with col2:
            if st.button("Export Summary Report"):
                summary = self.generate_summary_report(df)
                st.download_button(
                    label="Download Report",
                    data=summary,
                    file_name=f"financial_report_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )

        with col3:
            if st.button("Export Budget Analysis"):
                budget_analysis = self.generate_budget_analysis(df)
                st.download_button(
                    label="Download Analysis",
                    data=budget_analysis,
                    file_name=f"budget_analysis_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )

    def generate_summary_report(self, df: pd.DataFrame) -> str:
        """Generate a comprehensive summary report"""
        if df.empty:
            return "No data available for report generation."

        total_spending = df['amount'].sum()
        avg_transaction = df['amount'].mean()
        transaction_count = len(df)
        date_range = f"{df['date'].min()} to {df['date'].max()}"

        category_breakdown = df.groupby('category')['amount'].agg(['sum', 'count']).round(2)

        report = f"""
FINANCIAL SUMMARY REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATA OVERVIEW
=============
Date Range: {date_range}
Total Transactions: {transaction_count:,}
Total Spending: ${total_spending:,.2f}
Average Transaction: ${avg_transaction:.2f}

CATEGORY BREAKDOWN
==================
"""

        for category, data in category_breakdown.iterrows():
            percentage = (data['sum'] / total_spending) * 100
            report += f"{category}: ${data['sum']:,.2f} ({data['count']} transactions, {percentage:.1f}%)\n"

        # Top merchants
        top_merchants = df.groupby('merchant')['amount'].sum().nlargest(5)
        report += "\nTOP 5 MERCHANTS BY SPENDING\n"
        report += "===========================\n"
        for merchant, amount in top_merchants.items():
            report += f"{merchant}: ${amount:,.2f}\n"

        return report

    def generate_budget_analysis(self, df: pd.DataFrame) -> str:
        """Generate budget analysis report"""
        if df.empty:
            return "No data available for budget analysis."

        budget_targets = st.session_state.get('budget_targets', {})
        current_month_spending = df[df['date'].dt.month == datetime.now().month].groupby('category')['amount'].sum()

        analysis = f"""
BUDGET ANALYSIS REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CURRENT MONTH PERFORMANCE
=========================
"""

        for category, target in budget_targets.items():
            actual = current_month_spending.get(category, 0)
            percentage = (actual / target) * 100 if target > 0 else 0
            status = "OVER BUDGET" if percentage > 100 else "ON TRACK"

            analysis += f"""
{category}:
  Target: ${target:,.2f}
  Actual: ${actual:,.2f}
  Usage: {percentage:.1f}%
  Status: {status}
"""

        return analysis
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class ChartGenerator:
    def __init__(self):
        self.color_palette = {
            'Food & Dining': '#FF6B6B',
            'Transportation': '#4ECDC4',
            'Shopping': '#45B7D1',
            'Entertainment': '#96CEB4',
            'Bills & Utilities': '#FFEAA7'
        }

    def create_spending_overview(self, df: pd.DataFrame) -> go.Figure:
        """Create overview charts with key metrics"""
        df['date'] = pd.to_datetime(df['date'])

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Monthly Spending Trend', 'Category Breakdown',
                            'Daily Spending Pattern', 'Top Merchants'),
            specs=[[{"secondary_y": False}, {"type": "pie"}],
                   [{"secondary_y": False}, {"type": "bar"}]]
        )

        # Monthly trend
        monthly_data = df.groupby(df['date'].dt.to_period('M'))['amount'].sum().reset_index()
        monthly_data['date'] = monthly_data['date'].astype(str)

        fig.add_trace(
            go.Scatter(x=monthly_data['date'], y=monthly_data['amount'],
                       mode='lines+markers', name='Monthly Spending',
                       line=dict(color='#3498db', width=3)),
            row=1, col=1
        )

        # Category pie chart
        category_data = df.groupby('category')['amount'].sum().reset_index()
        colors = [self.color_palette.get(cat, '#BDC3C7') for cat in category_data['category']]

        fig.add_trace(
            go.Pie(labels=category_data['category'], values=category_data['amount'],
                   marker_colors=colors, name="Categories"),
            row=1, col=2
        )

        # Daily pattern
        daily_avg = df.groupby(df['date'].dt.day_name())['amount'].mean().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])

        fig.add_trace(
            go.Bar(x=daily_avg.index, y=daily_avg.values,
                   marker_color='#e74c3c', name='Avg Daily Spending'),
            row=2, col=1
        )

        # Top merchants
        top_merchants = df.groupby('merchant')['amount'].sum().nlargest(10)

        fig.add_trace(
            go.Bar(y=top_merchants.index, x=top_merchants.values,
                   orientation='h', marker_color='#9b59b6', name='Top Merchants'),
            row=2, col=2
        )

        fig.update_layout(height=800, showlegend=False, title_text="Spending Overview Dashboard")
        return fig

    def create_time_series_chart(self, df: pd.DataFrame, predictions: pd.DataFrame = None) -> go.Figure:
        """Create time series chart with predictions"""
        df['date'] = pd.to_datetime(df['date'])

        # Daily spending
        daily_spending = df.groupby('date')['amount'].sum().reset_index()

        fig = go.Figure()

        # Historical data
        fig.add_trace(go.Scatter(
            x=daily_spending['date'],
            y=daily_spending['amount'],
            mode='lines',
            name='Actual Spending',
            line=dict(color='#3498db', width=2)
        ))

        # Add 7-day moving average
        daily_spending['ma_7'] = daily_spending['amount'].rolling(window=7).mean()
        fig.add_trace(go.Scatter(
            x=daily_spending['date'],
            y=daily_spending['ma_7'],
            mode='lines',
            name='7-Day Average',
            line=dict(color='#e74c3c', width=2, dash='dash')
        ))

        # Add predictions if provided
        if predictions is not None and not predictions.empty:
            fig.add_trace(go.Scatter(
                x=predictions['date'],
                y=predictions['predicted_amount'],
                mode='lines+markers',
                name='Predicted Spending',
                line=dict(color='#f39c12', width=2, dash='dot'),
                marker=dict(size=6)
            ))

        fig.update_layout(
            title='Daily Spending Trends and Predictions',
            xaxis_title='Date',
            yaxis_title='Amount ($)',
            hovermode='x unified'
        )

        return fig

    def create_category_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Create detailed category analysis charts"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Category Spending Over Time', 'Average Transaction by Category',
                            'Transaction Count by Category', 'Spending Distribution'),
            specs=[[{"secondary_y": False}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "box"}]]
        )

        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M').astype(str)

        # Category spending over time
        monthly_category = df.groupby(['month', 'category'])['amount'].sum().reset_index()

        for category in df['category'].unique():
            cat_data = monthly_category[monthly_category['category'] == category]
            color = self.color_palette.get(category, '#BDC3C7')

            fig.add_trace(
                go.Scatter(x=cat_data['month'], y=cat_data['amount'],
                           name=category, line=dict(color=color, width=2)),
                row=1, col=1
            )

        # Average transaction by category
        avg_by_category = df.groupby('category')['amount'].mean().sort_values(ascending=True)
        colors = [self.color_palette.get(cat, '#BDC3C7') for cat in avg_by_category.index]

        fig.add_trace(
            go.Bar(y=avg_by_category.index, x=avg_by_category.values,
                   orientation='h', marker_color=colors, name='Avg Transaction'),
            row=1, col=2
        )

        # Transaction count
        count_by_category = df['category'].value_counts()
        colors = [self.color_palette.get(cat, '#BDC3C7') for cat in count_by_category.index]

        fig.add_trace(
            go.Bar(x=count_by_category.index, y=count_by_category.values,
                   marker_color=colors, name='Transaction Count'),
            row=2, col=1
        )

        # Box plot for spending distribution
        for category in df['category'].unique():
            cat_amounts = df[df['category'] == category]['amount']
            color = self.color_palette.get(category, '#BDC3C7')

            fig.add_trace(
                go.Box(y=cat_amounts, name=category, marker_color=color),
                row=2, col=2
            )

        fig.update_layout(height=800, showlegend=False, title_text="Category Analysis")
        return fig

    def create_anomaly_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create anomaly detection visualization"""
        df['date'] = pd.to_datetime(df['date'])

        fig = go.Figure()

        # Normal transactions
        normal_data = df[df.get('is_anomaly', 0) == 0]
        fig.add_trace(go.Scatter(
            x=normal_data['date'],
            y=normal_data['amount'],
            mode='markers',
            name='Normal Transactions',
            marker=dict(color='#3498db', size=6, opacity=0.6)
        ))

        # Anomalous transactions
        if 'is_anomaly' in df.columns:
            anomaly_data = df[df['is_anomaly'] == 1]
            if not anomaly_data.empty:
                fig.add_trace(go.Scatter(
                    x=anomaly_data['date'],
                    y=anomaly_data['amount'],
                    mode='markers',
                    name='Anomalous Transactions',
                    marker=dict(color='#e74c3c', size=10, symbol='x'),
                    text=anomaly_data['merchant'],
                    hovertemplate='<b>%{text}</b><br>Amount: $%{y}<br>Date: %{x}<extra></extra>'
                ))

        fig.update_layout(
            title='Transaction Anomaly Detection',
            xaxis_title='Date',
            yaxis_title='Amount ($)',
            hovermode='closest'
        )

        return fig
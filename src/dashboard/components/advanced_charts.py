import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st


class AdvancedChartGenerator:
    def __init__(self):
        self.color_palette = {
            'Food & Dining': '#FF6B6B',
            'Transportation': '#4ECDC4',
            'Shopping': '#45B7D1',
            'Entertainment': '#96CEB4',
            'Bills & Utilities': '#FFEAA7'
        }
        self.theme = {
            'background': '#FFFFFF',
            'grid': '#F0F0F0',
            'text': '#2C3E50'
        }

    def create_financial_health_dashboard(self, health_data: dict) -> go.Figure:
        """Create comprehensive financial health visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Overall Health Score', 'Component Breakdown',
                            'Health Trend', 'Improvement Areas'),
            specs=[[{"type": "indicator"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "pie"}]]
        )

        # Overall health score gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=health_data['overall_score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Financial Health"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self.get_health_color(health_data['overall_score'])},
                    'steps': [
                        {'range': [0, 50], 'color': '#FFE5E5'},
                        {'range': [50, 65], 'color': '#FFF4E5'},
                        {'range': [65, 80], 'color': '#E5F7E5'},
                        {'range': [80, 100], 'color': '#E5F5FF'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ),
            row=1, col=1
        )

        # Component breakdown
        components = health_data['components']
        fig.add_trace(
            go.Bar(
                x=list(components.keys()),
                y=list(components.values()),
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
                name="Components"
            ),
            row=1, col=2
        )

        fig.update_layout(height=800, showlegend=False, title_text="Financial Health Dashboard")
        return fig

    def get_health_color(self, score: float) -> str:
        """Get color based on health score"""
        if score >= 80:
            return "#27AE60"  # Green
        elif score >= 65:
            return "#F39C12"  # Orange
        elif score >= 50:
            return "#E74C3C"  # Red
        else:
            return "#95A5A6"  # Gray

    def create_predictive_analytics_chart(self, historical_data: pd.DataFrame,
                                          predictions: pd.DataFrame,
                                          confidence_intervals: pd.DataFrame = None) -> go.Figure:
        """Create advanced predictive analytics visualization"""
        fig = go.Figure()

        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_data['date'],
            y=historical_data['amount'],
            mode='lines+markers',
            name='Historical Spending',
            line=dict(color='#3498db', width=2),
            marker=dict(size=4)
        ))

        # Predictions
        fig.add_trace(go.Scatter(
            x=predictions['date'],
            y=predictions['predicted_amount'],
            mode='lines+markers',
            name='Predicted Spending',
            line=dict(color='#e74c3c', width=3, dash='dot'),
            marker=dict(size=6, symbol='diamond')
        ))

        # Confidence intervals if provided
        if confidence_intervals is not None:
            fig.add_trace(go.Scatter(
                x=predictions['date'],
                y=confidence_intervals['upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))

            fig.add_trace(go.Scatter(
                x=predictions['date'],
                y=confidence_intervals['lower'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(231, 76, 60, 0.2)',
                name='Confidence Interval',
                hoverinfo='skip'
            ))

        # Add prediction accuracy metrics
        if 'confidence' in predictions.columns:
            avg_confidence = predictions['confidence'].mean()
            fig.add_annotation(
                x=predictions['date'].iloc[-1],
                y=predictions['predicted_amount'].max(),
                text=f"Avg Confidence: {avg_confidence:.1%}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#636363",
                bgcolor="#FFFFFF",
                bordercolor="#636363",
                borderwidth=1
            )

        fig.update_layout(
            title='Predictive Spending Analytics with Confidence Intervals',
            xaxis_title='Date',
            yaxis_title='Amount ($)',
            hovermode='x unified',
            template='plotly_white'
        )

        return fig

    def create_behavioral_clustering_chart(self, cluster_data: dict) -> go.Figure:
        """Create spending behavior clustering visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cluster Sizes', 'Spending Patterns', 'Behavior Types', 'Characteristics'),
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "radar"}]]
        )

        # Extract cluster information
        cluster_names = list(cluster_data.keys())
        cluster_sizes = [cluster_data[name]['size'] for name in cluster_names]
        behavior_types = [cluster_data[name]['behavior_type'] for name in cluster_names]

        # Cluster sizes pie chart
        fig.add_trace(
            go.Pie(
                labels=behavior_types,
                values=cluster_sizes,
                name="Cluster Sizes"
            ),
            row=1, col=1
        )

        # Spending patterns scatter
        avg_spending = [cluster_data[name]['avg_daily_spending'] for name in cluster_names]
        avg_transactions = [cluster_data[name]['avg_transactions'] for name in cluster_names]

        fig.add_trace(
            go.Scatter(
                x=avg_transactions,
                y=avg_spending,
                mode='markers+text',
                text=behavior_types,
                textposition="top center",
                marker=dict(
                    size=[size / 10 for size in cluster_sizes],
                    color=avg_spending,
                    colorscale='Viridis',
                    showscale=True
                ),
                name="Spending Patterns"
            ),
            row=1, col=2
        )

        # Behavior types bar chart
        fig.add_trace(
            go.Bar(
                x=behavior_types,
                y=avg_spending,
                name="Avg Daily Spending",
                marker_color='lightblue'
            ),
            row=2, col=1
        )

        fig.update_layout(height=800, showlegend=False, title_text="Spending Behavior Analysis")
        return fig

    def create_efficiency_optimization_chart(self, efficiency_data: dict) -> go.Figure:
        """Create spending efficiency and optimization visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Category Efficiency', 'Merchant Analysis',
                            'Optimization Opportunities', 'Savings Potential'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "funnel"}, {"type": "indicator"}]]
        )

        # Category efficiency
        if 'category_efficiency' in efficiency_data:
            categories = list(efficiency_data['category_efficiency'].keys())
            efficiency_scores = [
                efficiency_data['category_efficiency'][cat]['efficiency_score']
                for cat in categories
            ]

            fig.add_trace(
                go.Bar(
                    x=categories,
                    y=efficiency_scores,
                    name="Efficiency Score",
                    marker_color='green'
                ),
                row=1, col=1
            )

        # Merchant analysis
        if 'merchant_analysis' in efficiency_data:
            merchants = list(efficiency_data['merchant_analysis'].keys())[:10]
            total_spent = [
                efficiency_data['merchant_analysis'][merchant]['total']
                for merchant in merchants
            ]
            frequency = [
                efficiency_data['merchant_analysis'][merchant]['count']
                for merchant in merchants
            ]

            fig.add_trace(
                go.Scatter(
                    x=frequency,
                    y=total_spent,
                    mode='markers+text',
                    text=merchants,
                    textposition="top center",
                    marker=dict(size=10, color='blue'),
                    name="Merchant Spending"
                ),
                row=1, col=2
            )

        # Optimization opportunities
        if 'optimization_opportunities' in efficiency_data:
            opportunities = efficiency_data['optimization_opportunities']
            if opportunities:
                opp_types = [opp['type'] for opp in opportunities]
                savings = [opp['potential_monthly_savings'] for opp in opportunities]

                fig.add_trace(
                    go.Funnel(
                        y=opp_types,
                        x=savings,
                        name="Savings Potential"
                    ),
                    row=2, col=1
                )

        fig.update_layout(height=800, showlegend=False, title_text="Spending Efficiency Analysis")
        return fig

    def create_real_time_alerts_chart(self, alerts_data: list) -> go.Figure:
        """Create real-time financial alerts visualization"""
        if not alerts_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No active alerts",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=20, color="gray")
            )
            return fig

        # Categorize alerts by severity
        severity_colors = {'high': '#e74c3c', 'medium': '#f39c12', 'low': '#3498db'}
        severity_counts = {}

        for alert in alerts_data:
            severity = alert.get('severity', 'low')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Alert Severity Distribution', 'Recent Alerts Timeline'),
            specs=[[{"type": "pie"}, {"type": "scatter"}]]
        )

        # Severity distribution
        fig.add_trace(
            go.Pie(
                labels=list(severity_counts.keys()),
                values=list(severity_counts.values()),
                marker_colors=[severity_colors[s] for s in severity_counts.keys()],
                name="Alert Severity"
            ),
            row=1, col=1
        )

        # Timeline of alerts (if timestamps available)
        alert_times = [i for i in range(len(alerts_data))]  # Placeholder for actual timestamps
        alert_severities = [alert.get('severity', 'low') for alert in alerts_data]
        alert_colors = [severity_colors[s] for s in alert_severities]

        fig.add_trace(
            go.Scatter(
                x=alert_times,
                y=[1] * len(alerts_data),
                mode='markers',
                marker=dict(
                    size=15,
                    color=alert_colors,
                    line=dict(width=2, color='white')
                ),
                text=[alert['message'] for alert in alerts_data],
                hovertemplate='<b>%{text}</b><extra></extra>',
                name="Alerts"
            ),
            row=1, col=2
        )

        fig.update_layout(height=400, title_text="Financial Alerts Dashboard")
        return fig

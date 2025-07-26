import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class InsightsGenerator:
    def __init__(self):
        self.insights = []

    def analyze_spending_trends(self, df: pd.DataFrame) -> List[Dict]:
        """Analyze spending trends and generate insights"""
        insights = []
        df['date'] = pd.to_datetime(df['date'])

        # Monthly spending trend
        monthly_spending = df.groupby(df['date'].dt.to_period('M'))['amount'].sum()
        if len(monthly_spending) >= 2:
            recent_change = ((monthly_spending.iloc[-1] - monthly_spending.iloc[-2]) /
                             monthly_spending.iloc[-2] * 100)

            if abs(recent_change) > 10:
                trend = "increased" if recent_change > 0 else "decreased"
                insights.append({
                    'type': 'spending_trend',
                    'title': f'Monthly Spending {trend.title()}',
                    'description': f'Your spending has {trend} by {abs(recent_change):.1f}% compared to last month',
                    'value': recent_change,
                    'priority': 'high' if abs(recent_change) > 20 else 'medium'
                })

        # Category analysis
        category_spending = df.groupby('category')['amount'].sum().sort_values(ascending=False)
        top_category = category_spending.index[0]
        top_category_pct = (category_spending.iloc[0] / category_spending.sum()) * 100

        if top_category_pct > 40:
            insights.append({
                'type': 'category_concentration',
                'title': f'High Spending in {top_category}',
                'description': f'{top_category} accounts for {top_category_pct:.1f}% of your total spending',
                'value': top_category_pct,
                'priority': 'medium'
            })

        return insights

    def analyze_savings_opportunities(self, df: pd.DataFrame) -> List[Dict]:
        """Identify potential savings opportunities"""
        insights = []
        df['date'] = pd.to_datetime(df['date'])

        # Frequent small purchases
        small_purchases = df[df['amount'] < 20]
        if len(small_purchases) > 0:
            weekly_small_purchases = len(small_purchases) / (
                    (df['date'].max() - df['date'].min()).days / 7
            )

            if weekly_small_purchases > 10:
                total_small = small_purchases['amount'].sum()
                insights.append({
                    'type': 'small_purchases',
                    'title': 'Frequent Small Purchases',
                    'description': f'You make {weekly_small_purchases:.1f} small purchases per week, totaling ${total_small:.2f}',
                    'value': total_small,
                    'priority': 'low'
                })

        # Weekend vs weekday spending
        df['is_weekend'] = df['date'].dt.dayofweek >= 5
        weekend_avg = df[df['is_weekend']]['amount'].mean()
        weekday_avg = df[~df['is_weekend']]['amount'].mean()

        if weekend_avg > weekday_avg * 1.5:
            insights.append({
                'type': 'weekend_spending',
                'title': 'Higher Weekend Spending',
                'description': f'Weekend transactions average ${weekend_avg:.2f} vs ${weekday_avg:.2f} on weekdays',
                'value': weekend_avg - weekday_avg,
                'priority': 'medium'
            })

        return insights

    def analyze_budget_performance(self, df: pd.DataFrame, budgets: Dict = None) -> List[Dict]:
        """Analyze budget performance if budgets are provided"""
        insights = []

        if not budgets:
            return insights

        current_month = datetime.now().month
        current_year = datetime.now().year

        # Filter current month data
        current_month_data = df[
            (df['date'].dt.month == current_month) &
            (df['date'].dt.year == current_year)
            ]

        # Category spending vs budget
        category_spending = current_month_data.groupby('category')['amount'].sum()

        for category, budget_limit in budgets.items():
            if category in category_spending.index:
                spent = category_spending[category]
                budget_usage = (spent / budget_limit) * 100

                if budget_usage > 90:
                    status = "over budget" if budget_usage > 100 else "approaching budget limit"
                    insights.append({
                        'type': 'budget_alert',
                        'title': f'{category} Budget Alert',
                        'description': f'You\'ve used {budget_usage:.1f}% of your {category} budget (${spent:.2f}/${budget_limit:.2f})',
                        'value': budget_usage,
                        'priority': 'high' if budget_usage > 100 else 'medium'
                    })

        return insights

    def generate_all_insights(self, df: pd.DataFrame, budgets: Dict = None) -> Dict:
        """Generate comprehensive insights"""
        logger.info("Generating financial insights")

        all_insights = []

        # Spending trends
        all_insights.extend(self.analyze_spending_trends(df))

        # Savings opportunities
        all_insights.extend(self.analyze_savings_opportunities(df))

        # Budget performance
        if budgets:
            all_insights.extend(self.analyze_budget_performance(df, budgets))

        # Sort by priority
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        all_insights.sort(key=lambda x: priority_order.get(x['priority'], 0), reverse=True)

        # Generate summary statistics
        total_spending = df['amount'].sum()
        avg_transaction = df['amount'].mean()
        transaction_count = len(df)

        summary = {
            'total_insights': len(all_insights),
            'high_priority_insights': len([i for i in all_insights if i['priority'] == 'high']),
            'spending_summary': {
                'total_spending': total_spending,
                'avg_transaction': avg_transaction,
                'transaction_count': transaction_count,
                'date_range': {
                    'start': df['date'].min(),
                    'end': df['date'].max()
                }
            },
            'insights': all_insights[:10]  # Top 10 insights
        }

        logger.info(f"Generated {len(all_insights)} insights")
        return summary
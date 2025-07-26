import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import logging
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AdvancedInsightsEngine:
    """ML-powered insights engine with behavioral analysis and personalized recommendations"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.spending_clusters = None
        self.behavioral_model = None

    def analyze_spending_behavior(self, df: pd.DataFrame) -> Dict:
        """Analyze spending behavior using clustering and statistical analysis"""
        # Create behavioral features
        behavioral_features = self.create_behavioral_features(df)

        # Cluster analysis
        cluster_insights = self.perform_cluster_analysis(behavioral_features)

        # Trend analysis
        trend_insights = self.analyze_spending_trends(df)

        # Efficiency analysis
        efficiency_insights = self.analyze_spending_efficiency(df)

        # Predictive insights
        predictive_insights = self.generate_predictive_insights(df)

        return {
            'behavioral_clusters': cluster_insights,
            'spending_trends': trend_insights,
            'efficiency_analysis': efficiency_insights,
            'predictive_insights': predictive_insights,
            'overall_score': self.calculate_financial_health_score(df)
        }

    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features that capture spending behavior patterns"""
        df['date'] = pd.to_datetime(df['date'])

        # Aggregate by date for behavioral analysis
        daily_spending = df.groupby('date').agg({
            'amount': ['sum', 'count', 'mean', 'std'],
            'category': lambda x: len(x.unique())
        }).fillna(0)

        daily_spending.columns = ['total_amount', 'transaction_count', 'avg_amount',
                                  'amount_std', 'category_diversity']

        # Add time-based features
        daily_spending['weekday'] = daily_spending.index.dayofweek
        daily_spending['is_weekend'] = (daily_spending['weekday'] >= 5).astype(int)
        daily_spending['month'] = daily_spending.index.month
        daily_spending['day_of_month'] = daily_spending.index.day

        # Behavioral patterns
        daily_spending['spending_consistency'] = (
                1 / (1 + daily_spending['amount_std'] / (daily_spending['avg_amount'] + 1))
        )

        # Impulse buying indicator
        daily_spending['impulse_indicator'] = (
                daily_spending['transaction_count'] * daily_spending['amount_std']
        )

        return daily_spending.fillna(0)

    def perform_cluster_analysis(self, behavioral_features: pd.DataFrame) -> Dict:
        """Perform clustering analysis on spending behavior"""
        # Select features for clustering
        cluster_features = ['total_amount', 'transaction_count', 'avg_amount',
                            'category_diversity', 'spending_consistency', 'impulse_indicator']

        X = behavioral_features[cluster_features]
        X_scaled = self.scaler.fit_transform(X)

        # Optimal number of clusters using elbow method
        optimal_k = self.find_optimal_clusters(X_scaled)

        # Perform clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        # Analyze clusters
        cluster_analysis = {}
        for i in range(optimal_k):
            cluster_data = behavioral_features[clusters == i]
            cluster_analysis[f'cluster_{i}'] = {
                'size': len(cluster_data),
                'avg_daily_spending': cluster_data['total_amount'].mean(),
                'avg_transactions': cluster_data['transaction_count'].mean(),
                'spending_consistency': cluster_data['spending_consistency'].mean(),
                'category_diversity': cluster_data['category_diversity'].mean(),
                'behavior_type': self.classify_behavior_type(cluster_data)
            }

        return cluster_analysis

    def find_optimal_clusters(self, X: np.ndarray, max_k: int = 8) -> int:
        """Find optimal number of clusters using elbow method"""
        inertias = []
        k_range = range(2, min(max_k + 1, len(X) // 2))

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        # Find elbow point
        if len(inertias) >= 2:
            diffs = np.diff(inertias)
            second_diffs = np.diff(diffs)
            if len(second_diffs) > 0:
                elbow_idx = np.argmax(second_diffs) + 2
                return min(k_range[elbow_idx], max_k)

        return 3  # Default fallback

    def classify_behavior_type(self, cluster_data: pd.DataFrame) -> str:
        """Classify spending behavior type based on cluster characteristics"""
        avg_spending = cluster_data['total_amount'].mean()
        consistency = cluster_data['spending_consistency'].mean()
        diversity = cluster_data['category_diversity'].mean()
        impulse = cluster_data['impulse_indicator'].mean()

        if consistency > 0.7 and impulse < cluster_data['impulse_indicator'].quantile(0.3):
            return "Conservative Spender"
        elif avg_spending > cluster_data['total_amount'].quantile(0.7) and diversity > 3:
            return "Diverse Spender"
        elif impulse > cluster_data['impulse_indicator'].quantile(0.7):
            return "Impulse Buyer"
        elif consistency > 0.5 and avg_spending < cluster_data['total_amount'].quantile(0.5):
            return "Budget Conscious"
        else:
            return "Moderate Spender"

    def analyze_spending_trends(self, df: pd.DataFrame) -> Dict:
        """Advanced trend analysis with statistical significance testing"""
        df['date'] = pd.to_datetime(df['date'])

        # Monthly trend analysis
        monthly_data = df.groupby(df['date'].dt.to_period('M'))['amount'].agg(['sum', 'count']).reset_index()
        monthly_data['date'] = monthly_data['date'].dt.to_timestamp()

        # Statistical trend test
        if len(monthly_data) >= 3:
            x = np.arange(len(monthly_data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, monthly_data['sum'])

            trend_significance = "significant" if p_value < 0.05 else "not significant"
            trend_direction = "increasing" if slope > 0 else "decreasing"

            # Seasonality detection
            seasonality = self.detect_seasonality(monthly_data)

            return {
                'monthly_trend': {
                    'slope': slope,
                    'direction': trend_direction,
                    'significance': trend_significance,
                    'r_squared': r_value ** 2,
                    'p_value': p_value
                },
                'seasonality': seasonality,
                'volatility': monthly_data['sum'].std() / monthly_data['sum'].mean()
            }

        return {'message': 'Insufficient data for trend analysis'}

    def detect_seasonality(self, monthly_data: pd.DataFrame) -> Dict:
        """Detect seasonal patterns in spending"""
        if len(monthly_data) >= 12:
            # Simple seasonal decomposition
            monthly_data['month'] = monthly_data['date'].dt.month
            seasonal_avg = monthly_data.groupby('month')['sum'].mean()

            peak_month = seasonal_avg.idxmax()
            low_month = seasonal_avg.idxmin()
            seasonal_strength = (seasonal_avg.max() - seasonal_avg.min()) / seasonal_avg.mean()

            return {
                'has_seasonality': seasonal_strength > 0.2,
                'peak_month': peak_month,
                'low_month': low_month,
                'seasonal_strength': seasonal_strength
            }

        return {'has_seasonality': False}

    def analyze_spending_efficiency(self, df: pd.DataFrame) -> Dict:
        """Analyze spending efficiency and identify optimization opportunities"""
        # Category efficiency analysis
        category_stats = df.groupby('category').agg({
            'amount': ['sum', 'count', 'mean', 'std']
        }).round(2)

        category_stats.columns = ['total', 'count', 'avg', 'std']
        category_stats['efficiency_score'] = (
                category_stats['total'] / (category_stats['std'] + 1)
        )

        # Merchant efficiency
        merchant_stats = df.groupby('merchant').agg({
            'amount': ['sum', 'count', 'mean']
        }).round(2)
        merchant_stats.columns = ['total', 'count', 'avg']

        # Find optimization opportunities
        optimization_opportunities = []

        # High-frequency low-value transactions
        small_frequent = df[(df['amount'] < 20) & (df['merchant'].isin(
            df['merchant'].value_counts().head(10).index
        ))]

        if len(small_frequent) > 0:
            weekly_small_spending = small_frequent.groupby(
                small_frequent['date'].dt.to_period('W')
            )['amount'].sum().mean()

            optimization_opportunities.append({
                'type': 'micro_transactions',
                'potential_monthly_savings': weekly_small_spending * 4 * 0.3,
                'description': 'Reduce frequent small purchases'
            })

        return {
            'category_efficiency': category_stats.to_dict('index'),
            'merchant_analysis': merchant_stats.head(10).to_dict('index'),
            'optimization_opportunities': optimization_opportunities
        }

    def generate_predictive_insights(self, df: pd.DataFrame) -> List[Dict]:
        """Generate predictive insights based on spending patterns"""
        insights = []
        df['date'] = pd.to_datetime(df['date'])

        # Predict budget overruns
        current_month = datetime.now().month
        current_month_data = df[df['date'].dt.month == current_month]

        if not current_month_data.empty:
            days_in_month = current_month_data['date'].dt.days_in_month.iloc[0]
            days_elapsed = datetime.now().day
            current_spending = current_month_data['amount'].sum()
            projected_monthly = (current_spending / days_elapsed) * days_in_month

            # Historical average for comparison
            historical_monthly = df.groupby(df['date'].dt.to_period('M'))['amount'].sum().mean()

            if projected_monthly > historical_monthly * 1.2:
                insights.append({
                    'type': 'budget_warning',
                    'severity': 'high',
                    'message': f'Projected monthly spending (${projected_monthly:.2f}) is 20% above historical average',
                    'recommendation': 'Consider reducing discretionary spending this month'
                })

        # Category trend predictions
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]
            monthly_cat = cat_data.groupby(cat_data['date'].dt.to_period('M'))['amount'].sum()

            if len(monthly_cat) >= 3:
                recent_trend = monthly_cat.tail(3).pct_change().mean()
                if abs(recent_trend) > 0.15:  # 15% change
                    direction = "increasing" if recent_trend > 0 else "decreasing"
                    insights.append({
                        'type': 'category_trend',
                        'severity': 'medium',
                        'message': f'{category} spending is {direction} by {abs(recent_trend) * 100:.1f}%',
                        'recommendation': f'Monitor {category} expenses closely'
                    })

        return insights

    def calculate_financial_health_score(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive financial health score"""
        # Various health indicators
        spending_consistency = self.calculate_consistency_score(df)
        category_balance = self.calculate_category_balance_score(df)
        trend_stability = self.calculate_trend_stability_score(df)
        efficiency_score = self.calculate_efficiency_score(df)

        # Weighted overall score
        overall_score = (
                                spending_consistency * 0.25 +
                                category_balance * 0.25 +
                                trend_stability * 0.25 +
                                efficiency_score * 0.25
                        ) * 100

        # Health level classification
        if overall_score >= 80:
            health_level = "Excellent"
        elif overall_score >= 65:
            health_level = "Good"
        elif overall_score >= 50:
            health_level = "Fair"
        else:
            health_level = "Needs Improvement"

        return {
            'overall_score': round(overall_score, 1),
            'health_level': health_level,
            'components': {
                'spending_consistency': round(spending_consistency * 100, 1),
                'category_balance': round(category_balance * 100, 1),
                'trend_stability': round(trend_stability * 100, 1),
                'efficiency': round(efficiency_score * 100, 1)
            }
        }

    def calculate_consistency_score(self, df: pd.DataFrame) -> float:
        """Calculate spending consistency score"""
        daily_spending = df.groupby(df['date'])['amount'].sum()
        cv = daily_spending.std() / daily_spending.mean() if daily_spending.mean() > 0 else 1
        return max(0, 1 - cv / 2)  # Lower coefficient of variation = higher score

    def calculate_category_balance_score(self, df: pd.DataFrame) -> float:
        """Calculate category balance score"""
        category_dist = df.groupby('category')['amount'].sum()
        # Penalize extreme concentration in single category
        max_category_pct = category_dist.max() / category_dist.sum()
        return max(0, 1 - max_category_pct) if max_category_pct < 0.8 else 0.2

    def calculate_trend_stability_score(self, df: pd.DataFrame) -> float:
        """Calculate trend stability score"""
        monthly_spending = df.groupby(df['date'].dt.to_period('M'))['amount'].sum()
        if len(monthly_spending) < 3:
            return 0.5

        # Calculate trend volatility
        pct_changes = monthly_spending.pct_change().dropna()
        volatility = pct_changes.std()
        return max(0, 1 - volatility * 2)

    def calculate_efficiency_score(self, df: pd.DataFrame) -> float:
        """Calculate spending efficiency score"""
        # Based on frequency of small transactions and merchant diversity
        small_transactions = len(df[df['amount'] < 10]) / len(df)
        merchant_diversity = len(df['merchant'].unique()) / len(df) * 10  # Normalized

        efficiency = 1 - small_transactions * 0.5 + min(merchant_diversity, 0.3)
        return max(0, min(1, efficiency))
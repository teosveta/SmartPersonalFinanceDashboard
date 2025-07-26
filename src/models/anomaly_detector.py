import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class AnomalyDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.category_stats = {}
        self.is_trained = False

    def calculate_statistical_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using statistical methods"""
        df_anomaly = df.copy()

        # Z-score based anomalies (overall)
        df_anomaly['amount_zscore'] = np.abs(stats.zscore(df_anomaly['amount']))
        df_anomaly['is_amount_outlier'] = (df_anomaly['amount_zscore'] > 3).astype(int)

        # Category-specific anomalies
        category_anomalies = []
        for category in df_anomaly['category'].unique():
            cat_data = df_anomaly[df_anomaly['category'] == category]['amount']
            cat_mean = cat_data.mean()
            cat_std = cat_data.std()

            # Store category statistics
            self.category_stats[category] = {
                'mean': cat_mean,
                'std': cat_std,
                'q1': cat_data.quantile(0.25),
                'q3': cat_data.quantile(0.75)
            }

            # Calculate IQR-based outliers
            iqr = self.category_stats[category]['q3'] - self.category_stats[category]['q1']
            lower_bound = self.category_stats[category]['q1'] - 1.5 * iqr
            upper_bound = self.category_stats[category]['q3'] + 1.5 * iqr

            cat_outliers = (
                    (df_anomaly['category'] == category) &
                    ((df_anomaly['amount'] < lower_bound) | (df_anomaly['amount'] > upper_bound))
            )
            category_anomalies.append(cat_outliers)

        # Combine category anomalies
        df_anomaly['is_category_outlier'] = np.any(category_anomalies, axis=0).astype(int)

        return df_anomaly

    def detect_temporal_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect temporal spending anomalies"""
        df_temporal = df.copy()
        df_temporal['date'] = pd.to_datetime(df_temporal['date'])

        # Daily spending totals
        daily_spending = df_temporal.groupby('date')['amount'].sum().reset_index()
        daily_mean = daily_spending['amount'].mean()
        daily_std = daily_spending['amount'].std()

        # Identify unusual spending days
        daily_spending['is_high_spend_day'] = (
                daily_spending['amount'] > daily_mean + 2 * daily_std
        ).astype(int)

        # Merge back to original data
        df_temporal = df_temporal.merge(
            daily_spending[['date', 'is_high_spend_day']],
            on='date',
            how='left'
        )

        # Weekend vs weekday anomalies
        df_temporal['weekday'] = df_temporal['date'].dt.dayofweek
        df_temporal['is_weekend'] = (df_temporal['weekday'] >= 5).astype(int)

        # Calculate weekend/weekday spending patterns
        weekend_spending = df_temporal[df_temporal['is_weekend'] == 1]['amount']
        weekday_spending = df_temporal[df_temporal['is_weekend'] == 0]['amount']

        weekend_mean = weekend_spending.mean() if len(weekend_spending) > 0 else 0
        weekday_mean = weekday_spending.mean() if len(weekday_spending) > 0 else 0

        # Flag unusual weekend/weekday spending
        df_temporal['unusual_weekend_spending'] = (
                (df_temporal['is_weekend'] == 1) &
                (df_temporal['amount'] > weekend_mean + 2 * weekend_spending.std())
        ).astype(int)

        df_temporal['unusual_weekday_spending'] = (
                (df_temporal['is_weekend'] == 0) &
                (df_temporal['amount'] > weekday_mean + 2 * weekday_spending.std())
        ).astype(int)

        return df_temporal

    def train_isolation_forest(self, df: pd.DataFrame) -> Dict:
        """Train isolation forest for anomaly detection"""
        # Select numerical features for isolation forest
        feature_cols = ['amount', 'weekday', 'month', 'day']

        # Add encoded categorical features if available
        if 'category' in df.columns:
            category_encoded = pd.get_dummies(df['category'], prefix='cat')
            df_features = pd.concat([df[feature_cols], category_encoded], axis=1)
        else:
            df_features = df[feature_cols]

        # Handle missing values
        df_features = df_features.fillna(df_features.mean())

        # Scale features
        X_scaled = self.scaler.fit_transform(df_features)

        # Train isolation forest
        self.isolation_forest.fit(X_scaled)

        # Get anomaly scores
        anomaly_scores = self.isolation_forest.decision_function(X_scaled)
        predictions = self.isolation_forest.predict(X_scaled)

        self.is_trained = True

        # Calculate metrics
        n_anomalies = sum(predictions == -1)
        anomaly_rate = n_anomalies / len(predictions)

        logger.info(f"Isolation Forest trained - Anomaly rate: {anomaly_rate:.3f}")

        return {
            'anomaly_rate': anomaly_rate,
            'n_anomalies': n_anomalies,
            'total_samples': len(predictions)
        }

    def detect_all_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect all types of anomalies"""
        logger.info("Starting comprehensive anomaly detection")

        # Statistical anomalies
        df_result = self.calculate_statistical_anomalies(df)

        # Temporal anomalies
        df_result = self.detect_temporal_anomalies(df_result)

        # ML-based anomalies
        if self.is_trained:
            feature_cols = ['amount', 'weekday', 'month', 'day']
            df_features = df_result[feature_cols].fillna(df_result[feature_cols].mean())

            # Add category encoding if available
            if 'category' in df_result.columns:
                category_encoded = pd.get_dummies(df_result['category'], prefix='cat')
                df_features = pd.concat([df_features, category_encoded], axis=1)

            X_scaled = self.scaler.transform(df_features)
            ml_predictions = self.isolation_forest.predict(X_scaled)
            df_result['is_ml_anomaly'] = (ml_predictions == -1).astype(int)
        else:
            df_result['is_ml_anomaly'] = 0

        # Create composite anomaly score
        anomaly_columns = [
            'is_amount_outlier', 'is_category_outlier', 'is_high_spend_day',
            'unusual_weekend_spending', 'unusual_weekday_spending', 'is_ml_anomaly'
        ]

        df_result['anomaly_score'] = df_result[anomaly_columns].sum(axis=1)
        df_result['is_anomaly'] = (df_result['anomaly_score'] >= 2).astype(int)

        n_anomalies = df_result['is_anomaly'].sum()
        logger.info(f"Detected {n_anomalies} anomalies out of {len(df_result)} transactions")

        return df_result

    def get_anomaly_summary(self, df_anomalies: pd.DataFrame) -> Dict:
        """Generate summary of detected anomalies"""
        anomalies = df_anomalies[df_anomalies['is_anomaly'] == 1]

        if len(anomalies) == 0:
            return {'message': 'No significant anomalies detected'}

        summary = {
            'total_anomalies': len(anomalies),
            'anomaly_rate': len(anomalies) / len(df_anomalies),
            'avg_anomaly_amount': anomalies['amount'].mean(),
            'max_anomaly_amount': anomalies['amount'].max(),
            'anomaly_by_category': anomalies['category'].value_counts().to_dict(),
            'anomaly_types': {
                'amount_outliers': anomalies['is_amount_outlier'].sum(),
                'category_outliers': anomalies['is_category_outlier'].sum(),
                'high_spend_days': anomalies['is_high_spend_day'].sum(),
                'ml_detected': anomalies['is_ml_anomaly'].sum()
            },
            'recent_anomalies': anomalies.nlargest(5, 'amount')[
                ['date', 'amount', 'category', 'merchant', 'anomaly_score']
            ].to_dict('records')
        }

        return summary
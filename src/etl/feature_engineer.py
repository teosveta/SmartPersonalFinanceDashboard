# src/etl/feature_engineer.py (Complete implementation)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self):
        self.feature_columns = []

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Basic time features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['weekday'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
        df['day_of_year'] = df['date'].dt.dayofyear

        # Cyclical encoding for periodic features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)

        # Special periods
        df['is_holiday_season'] = ((df['month'] == 12) | (df['month'] == 1)).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)

        logger.info("Created time-based features")
        return df

    def create_spending_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create spending pattern features"""
        df = df.copy()
        df = df.sort_values('date')

        # Rolling averages
        df['amount_7d_avg'] = df['amount'].rolling(window=7, min_periods=1).mean()
        df['amount_30d_avg'] = df['amount'].rolling(window=30, min_periods=1).mean()
        df['amount_90d_avg'] = df['amount'].rolling(window=90, min_periods=1).mean()

        # Rolling sums
        df['amount_7d_sum'] = df['amount'].rolling(window=7, min_periods=1).sum()
        df['amount_30d_sum'] = df['amount'].rolling(window=30, min_periods=1).sum()

        # Cumulative features
        df['cumulative_spending'] = df['amount'].cumsum()
        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
        df['avg_daily_spending'] = df['cumulative_spending'] / (df['days_since_start'] + 1)

        # Category-specific features
        category_features = df.groupby('category')['amount'].transform('mean')
        df['amount_vs_category_avg'] = df['amount'] / category_features

        # Merchant frequency
        merchant_counts = df['merchant'].value_counts()
        df['merchant_frequency'] = df['merchant'].map(merchant_counts)

        logger.info("Created spending pattern features")
        return df

    def create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create categorical and text-based features"""
        df = df.copy()

        # Payment method encoding
        payment_dummies = pd.get_dummies(df['payment_method'], prefix='payment')
        df = pd.concat([df, payment_dummies], axis=1)

        # Category encoding
        category_dummies = pd.get_dummies(df['category'], prefix='category')
        df = pd.concat([df, category_dummies], axis=1)

        # Amount bins
        df['amount_bin'] = pd.cut(df['amount'],
                                  bins=[0, 20, 50, 100, 200, float('inf')],
                                  labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        amount_bin_dummies = pd.get_dummies(df['amount_bin'], prefix='amount_bin')
        df = pd.concat([df, amount_bin_dummies], axis=1)

        # Description length and features
        df['description_length'] = df['description'].str.len()
        df['has_online_keyword'] = df['description'].str.contains(
            'online|website|app|digital', case=False, na=False
        ).astype(int)

        logger.info("Created categorical features")
        return df

    def create_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for anomaly detection"""
        df = df.copy()

        # Z-score for amount (overall and by category)
        df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()

        category_stats = df.groupby('category')['amount'].agg(['mean', 'std'])
        df = df.merge(category_stats, left_on='category', right_index=True, suffixes=('', '_cat'))
        df['amount_zscore_category'] = (df['amount'] - df['mean']) / df['std']

        # Time since last transaction at same merchant
        df_sorted = df.sort_values(['merchant', 'date'])
        df_sorted['days_since_last_merchant'] = df_sorted.groupby('merchant')['date'].diff().dt.days
        df = df.merge(df_sorted[['merchant', 'date', 'days_since_last_merchant']],
                      on=['merchant', 'date'], how='left')

        # Unusual spending for day of week
        dow_avg = df.groupby(['category', 'weekday'])['amount'].mean().reset_index()
        dow_avg.columns = ['category', 'weekday', 'dow_category_avg']
        df = df.merge(dow_avg, on=['category', 'weekday'], how='left')
        df['amount_vs_dow_avg'] = df['amount'] / df['dow_category_avg']

        logger.info("Created anomaly detection features")
        return df

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features in proper order"""
        logger.info("Starting feature engineering process")

        # Create features in order (some depend on others)
        df = self.create_time_features(df)
        df = self.create_spending_patterns(df)
        df = self.create_categorical_features(df)
        df = self.create_anomaly_features(df)

        # Store feature column names for later use
        self.feature_columns = [col for col in df.columns if col not in [
            'id', 'date', 'amount', 'category', 'subcategory',
            'merchant', 'description', 'payment_method', 'created_at'
        ]]

        logger.info(f"Feature engineering complete. Created {len(self.feature_columns)} features")
        return df

    def get_feature_importance_data(self, df: pd.DataFrame) -> Dict:
        """Prepare data for feature importance analysis"""
        numeric_features = df.select_dtypes(include=[np.number]).columns
        categorical_features = df.select_dtypes(include=['object', 'category']).columns

        return {
            'numeric_features': numeric_features.tolist(),
            'categorical_features': categorical_features.tolist(),
            'total_features': len(self.feature_columns),
            'feature_types': {
                'time_based': [col for col in self.feature_columns if
                               any(x in col for x in ['month', 'day', 'year', 'weekend', 'holiday'])],
                'spending_patterns': [col for col in self.feature_columns if
                                      any(x in col for x in ['avg', 'sum', 'cumulative', 'frequency'])],
                'categorical': [col for col in self.feature_columns if
                                any(x in col for x in ['payment_', 'category_', 'amount_bin'])],
                'anomaly': [col for col in self.feature_columns if
                            any(x in col for x in ['zscore', 'since_last', 'vs_'])]
            }
        }

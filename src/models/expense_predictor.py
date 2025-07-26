import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import joblib
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ExpensePredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.is_trained = False

    def prepare_time_series_data(self, df: pd.DataFrame, category: str = None) -> pd.DataFrame:
        """Prepare data for time series forecasting"""
        df_ts = df.copy()
        df_ts['date'] = pd.to_datetime(df_ts['date'])

        if category:
            df_ts = df_ts[df_ts['category'] == category]

        # Aggregate by date
        daily_spending = df_ts.groupby('date')['amount'].sum().reset_index()

        # Create complete date range
        date_range = pd.date_range(
            start=daily_spending['date'].min(),
            end=daily_spending['date'].max(),
            freq='D'
        )

        # Reindex to include missing dates with 0 spending
        daily_spending = daily_spending.set_index('date').reindex(date_range, fill_value=0)
        daily_spending.index.name = 'date'
        daily_spending = daily_spending.reset_index()

        return daily_spending

    def create_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 7, 14, 30]) -> pd.DataFrame:
        """Create lagged features for ML prediction"""
        df_features = df.copy()

        for lag in lags:
            df_features[f'amount_lag_{lag}'] = df_features['amount'].shift(lag)

        # Rolling features
        df_features['amount_roll_7'] = df_features['amount'].rolling(window=7).mean()
        df_features['amount_roll_30'] = df_features['amount'].rolling(window=30).mean()
        df_features['amount_roll_std_7'] = df_features['amount'].rolling(window=7).std()

        # Time features
        df_features['dayofweek'] = df_features['date'].dt.dayofweek
        df_features['month'] = df_features['date'].dt.month
        df_features['day'] = df_features['date'].dt.day
        df_features['is_weekend'] = (df_features['dayofweek'] >= 5).astype(int)
        df_features['is_month_end'] = (df_features['date'].dt.day >= 25).astype(int)

        # Cyclical encoding
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        df_features['day_sin'] = np.sin(2 * np.pi * df_features['dayofweek'] / 7)
        df_features['day_cos'] = np.cos(2 * np.pi * df_features['dayofweek'] / 7)

        return df_features

    def train_ml_model(self, df: pd.DataFrame, category: str = None) -> Dict:
        """Train ML model for expense prediction"""
        ts_data = self.prepare_time_series_data(df, category)
        feature_data = self.create_lag_features(ts_data)

        # Remove rows with NaN values
        feature_data = feature_data.dropna()

        if len(feature_data) < 50:
            logger.warning(f"Not enough data for training ML model: {len(feature_data)} samples")
            return {}

        # Prepare features and target
        feature_cols = [col for col in feature_data.columns if col not in ['date', 'amount']]
        X = feature_data[feature_cols]
        y = feature_data['amount']

        # Split data (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Predictions and metrics
        y_pred = rf_model.predict(X_test)
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }

        # Store model
        model_key = f"ml_{category}" if category else "ml_total"
        self.models[model_key] = rf_model
        self.feature_columns = feature_cols

        logger.info(f"ML model trained for {model_key} - R²: {metrics['r2']:.3f}")
        return metrics

    def predict_future_expenses(self, df: pd.DataFrame, days_ahead: int = 30,
                                category: str = None) -> pd.DataFrame:
        """Predict future expenses"""
        model_key = f"ml_{category}" if category else "ml_total"

        if model_key not in self.models:
            logger.error(f"Model {model_key} not trained")
            return pd.DataFrame()

        model = self.models[model_key]
        ts_data = self.prepare_time_series_data(df, category)

        # Get last date and create future dates
        last_date = ts_data['date'].max()
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=days_ahead,
            freq='D'
        )

        # Prepare recent data for feature engineering
        recent_data = ts_data.tail(60).copy()  # Use last 60 days for context

        predictions = []
        current_data = recent_data.copy()

        for future_date in future_dates:
            # Add future date row
            new_row = pd.DataFrame({
                'date': [future_date],
                'amount': [0]  # Placeholder
            })
            current_data = pd.concat([current_data, new_row], ignore_index=True)

            # Create features
            feature_data = self.create_lag_features(current_data.tail(40))
            feature_row = feature_data.tail(1)[self.feature_columns]

            # Handle any NaN values
            feature_row = feature_row.fillna(feature_row.mean())

            # Predict
            prediction = model.predict(feature_row)[0]
            predictions.append({
                'date': future_date,
                'predicted_amount': max(0, prediction),  # Ensure non-negative
                'category': category
            })

            # Update the amount for next iteration
            current_data.iloc[-1, current_data.columns.get_loc('amount')] = prediction

        return pd.DataFrame(predictions)
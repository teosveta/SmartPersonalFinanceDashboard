import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import xgboost as xgb
import joblib
import logging
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class EnsembleExpensePredictor:
    """Advanced ensemble predictor with multiple algorithms and hyperparameter tuning"""

    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=150, max_depth=8, random_state=42),
            'xgb': xgb.XGBRegressor(n_estimators=200, max_depth=10, random_state=42),
            'linear': LinearRegression()
        }
        self.ensemble_weights = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.is_trained = False

    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sophisticated financial features"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Advanced time features
        df['quarter'] = df['date'].dt.quarter
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['is_month_start'] = (df['date'].dt.day <= 5).astype(int)
        df['is_month_end'] = (df['date'].dt.day >= 25).astype(int)
        df['is_payday'] = ((df['date'].dt.day == 15) | (df['date'].dt.day == 30)).astype(int)

        # Economic cycle features
        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
        df['economic_cycle'] = np.sin(2 * np.pi * df['days_since_start'] / 365.25)

        # Advanced rolling statistics
        for window in [3, 7, 14, 30, 90]:
            df[f'amount_rolling_mean_{window}'] = df['amount'].rolling(window=window, min_periods=1).mean()
            df[f'amount_rolling_std_{window}'] = df['amount'].rolling(window=window, min_periods=1).std()
            df[f'amount_rolling_median_{window}'] = df['amount'].rolling(window=window, min_periods=1).median()
            df[f'amount_rolling_max_{window}'] = df['amount'].rolling(window=window, min_periods=1).max()
            df[f'amount_rolling_min_{window}'] = df['amount'].rolling(window=window, min_periods=1).min()

        # Velocity and acceleration features
        df['spending_velocity'] = df['amount'].diff()
        df['spending_acceleration'] = df['spending_velocity'].diff()

        # Category momentum
        for category in df['category'].unique():
            cat_mask = df['category'] == category
            df.loc[cat_mask, f'category_{category}_momentum'] = (
                df.loc[cat_mask, 'amount'].rolling(window=7, min_periods=1).mean().diff()
            )

        # Merchant loyalty features
        merchant_stats = df.groupby('merchant').agg({
            'amount': ['count', 'mean', 'std'],
            'date': ['min', 'max']
        }).fillna(0)
        merchant_stats.columns = ['merchant_frequency', 'merchant_avg_amount',
                                  'merchant_amount_std', 'merchant_first_date', 'merchant_last_date']

        df = df.merge(merchant_stats, left_on='merchant', right_index=True, how='left')
        df['merchant_loyalty_days'] = (df['date'] - df['merchant_first_date']).dt.days

        # Seasonal decomposition features
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofyear / 365.25)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofyear / 365.25)

        return df.fillna(0)

    def train_with_cross_validation(self, df: pd.DataFrame, target_col: str = 'amount') -> Dict:
        """Train ensemble with time series cross-validation"""
        feature_df = self.create_advanced_features(df)

        # Select features (exclude non-predictive columns)
        exclude_cols = ['id', 'date', 'merchant', 'description', 'created_at', 'updated_at',
                        'merchant_first_date', 'merchant_last_date', target_col]
        feature_cols = [col for col in feature_df.columns if col not in exclude_cols]

        X = feature_df[feature_cols].select_dtypes(include=[np.number])
        y = feature_df[target_col]

        # Handle infinite values and NaN
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        model_scores = {}

        for name, model in self.models.items():
            try:
                scores = cross_val_score(model, X_scaled, y, cv=tscv,
                                         scoring='neg_mean_absolute_error', n_jobs=-1)
                model_scores[name] = -scores.mean()
                logger.info(f"{name} CV MAE: {model_scores[name]:.2f}")
            except Exception as e:
                logger.warning(f"Error training {name}: {e}")
                model_scores[name] = float('inf')

        # Train final models and calculate ensemble weights
        for name, model in self.models.items():
            if model_scores[name] != float('inf'):
                model.fit(X_scaled, y)

        # Calculate ensemble weights based on inverse of CV scores
        total_inverse_score = sum(1 / score for score in model_scores.values() if score != float('inf'))
        self.ensemble_weights = {
            name: (1 / score) / total_inverse_score
            for name, score in model_scores.items()
            if score != float('inf')
        }

        # Feature importance analysis
        self.analyze_feature_importance(X.columns)

        self.feature_columns = X.columns.tolist()
        self.is_trained = True

        return {
            'cv_scores': model_scores,
            'ensemble_weights': self.ensemble_weights,
            'feature_count': len(self.feature_columns)
        }

    def analyze_feature_importance(self, feature_names):
        """Analyze feature importance across models"""
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(feature_names, model.feature_importances_))
                self.feature_importance[name] = sorted(importance.items(),
                                                       key=lambda x: x[1], reverse=True)[:20]

    def predict_ensemble(self, df: pd.DataFrame, days_ahead: int = 30) -> pd.DataFrame:
        """Make ensemble predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        feature_df = self.create_advanced_features(df)
        X = feature_df[self.feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = self.scaler.transform(X)

        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            if name in self.ensemble_weights:
                try:
                    pred = model.predict(X_scaled)
                    predictions[name] = pred
                except Exception as e:
                    logger.warning(f"Prediction error for {name}: {e}")

        # Ensemble prediction
        if predictions:
            ensemble_pred = np.zeros(len(X))
            for name, pred in predictions.items():
                ensemble_pred += pred * self.ensemble_weights[name]

            return pd.DataFrame({
                'date': feature_df['date'],
                'predicted_amount': np.maximum(ensemble_pred, 0),  # Ensure non-negative
                'confidence': self.calculate_prediction_confidence(predictions)
            })

        return pd.DataFrame()

    def calculate_prediction_confidence(self, predictions: Dict) -> np.ndarray:
        """Calculate prediction confidence based on model agreement"""
        if len(predictions) < 2:
            return np.ones(len(list(predictions.values())[0])) * 0.5

        pred_array = np.array(list(predictions.values()))
        # Confidence based on inverse of standard deviation across models
        std_dev = np.std(pred_array, axis=0)
        confidence = 1 / (1 + std_dev / np.mean(pred_array, axis=0))
        return np.clip(confidence, 0.1, 0.95)
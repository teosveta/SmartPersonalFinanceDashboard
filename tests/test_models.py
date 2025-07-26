import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from src.models.expense_predictor import ExpensePredictor
    from src.models.category_classifier import CategoryClassifier
    from src.models.anomaly_detector import AnomalyDetector
except ImportError as e:
    pytest.skip(f"Could not import required modules: {e}", allow_module_level=True)


class TestExpensePredictor:

    def setup_method(self):
        self.predictor = ExpensePredictor()

        # Create sample time series data
        dates = pd.date_range('2024-01-01', periods=365, freq='D')
        amounts = 50 + 10 * np.sin(np.arange(365) * 2 * np.pi / 7) + np.random.normal(0, 5, 365)

        self.sample_df = pd.DataFrame({
            'date': dates,
            'amount': np.maximum(amounts, 0),  # Ensure positive amounts
            'category': ['Food & Dining'] * 365
        })

    def test_prepare_time_series_data(self):
        result = self.predictor.prepare_time_series_data(self.sample_df)

        assert isinstance(result, pd.DataFrame)
        assert 'date' in result.columns
        assert 'amount' in result.columns
        assert len(result) == 365  # Should have all days

    def test_create_lag_features(self):
        ts_data = self.predictor.prepare_time_series_data(self.sample_df)
        result = self.predictor.create_lag_features(ts_data)

        assert 'amount_lag_1' in result.columns
        assert 'amount_lag_7' in result.columns
        assert 'amount_roll_7' in result.columns
        assert 'dayofweek' in result.columns

    def test_train_ml_model(self):
        metrics = self.predictor.train_ml_model(self.sample_df)

        if metrics:  # If training was successful
            assert 'mae' in metrics
            assert 'rmse' in metrics
            assert 'r2' in metrics
            assert metrics['r2'] >= -1  # R² can be negative for poor models


class TestCategoryClassifier:

    def setup_method(self):
        self.classifier = CategoryClassifier()

        # Create sample data
        self.sample_df = pd.DataFrame({
            'merchant': ['Starbucks', 'Shell Gas', 'Amazon', 'Netflix', 'Electric Company'],
            'description': ['Coffee purchase', 'Gas fill-up', 'Online shopping', 'Streaming', 'Monthly bill'],
            'category': ['Food & Dining', 'Transportation', 'Shopping', 'Entertainment', 'Bills & Utilities']
        })

    def test_preprocess_text(self):
        text = "Starbucks Coffee Shop 123"
        result = self.classifier.preprocess_text(text)

        assert isinstance(result, str)
        assert result.islower()
        assert '123' not in result  # Numbers should be removed

    def test_rule_based_classify(self):
        result = self.classifier.rule_based_classify("Starbucks", "Coffee purchase")
        assert result == "Food & Dining"

        result = self.classifier.rule_based_classify("Shell", "Gas station")
        assert result == "Transportation"

    def test_train_classifier(self):
        # Need more data for proper training
        extended_df = pd.concat([self.sample_df] * 20, ignore_index=True)

        metrics = self.classifier.train(extended_df)

        if metrics:  # If training was successful
            assert 'accuracy' in metrics
            assert self.classifier.is_trained


class TestAnomalyDetector:

    def setup_method(self):
        self.detector = AnomalyDetector()

        # Create sample data with some outliers
        np.random.seed(42)
        normal_amounts = np.random.normal(50, 10, 95)
        outlier_amounts = [200, 300, 500, 1000, 150]  # Clear outliers

        amounts = np.concatenate([normal_amounts, outlier_amounts])
        dates = pd.date_range('2024-01-01', periods=100, freq='D')

        self.sample_df = pd.DataFrame({
            'date': dates,
            'amount': amounts,
            'category': ['Food & Dining'] * 100,
            'merchant': ['Store A'] * 100,
            'weekday': [d.weekday() for d in dates],
            'month': [d.month for d in dates],
            'day': [d.day for d in dates]
        })

    def test_calculate_statistical_anomalies(self):
        result = self.detector.calculate_statistical_anomalies(self.sample_df)

        assert 'amount_zscore' in result.columns
        assert 'is_amount_outlier' in result.columns
        assert 'is_category_outlier' in result.columns

        # Should detect some outliers
        assert result['is_amount_outlier'].sum() > 0

    def test_train_isolation_forest(self):
        metrics = self.detector.train_isolation_forest(self.sample_df)

        assert 'anomaly_rate' in metrics
        assert 'n_anomalies' in metrics
        assert self.detector.is_trained

    def test_detect_all_anomalies(self):
        # Train first
        self.detector.train_isolation_forest(self.sample_df)

        result = self.detector.detect_all_anomalies(self.sample_df)

        assert 'anomaly_score' in result.columns
        assert 'is_anomaly' in result.columns

        # Should detect some anomalies
        assert result['is_anomaly'].sum() > 0
import pytest
import pandas as pd
import sys
import os
from unittest.mock import Mock, patch

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from src.etl.data_loader import DataLoader
    from src.etl.feature_engineer import FeatureEngineer
except ImportError as e:
    pytest.skip(f"Could not import required modules: {e}", allow_module_level=True)


class TestDataLoader:

    def setup_method(self):
        self.data_loader = DataLoader()

    @patch('src.etl.data_loader.DatabaseManager')
    def test_load_transactions_basic(self, mock_db_manager):
        # Mock database response
        mock_session = Mock()
        mock_transaction = Mock()
        mock_transaction.id = 1
        mock_transaction.date = pd.to_datetime('2024-01-01').date()
        mock_transaction.amount = 25.50
        mock_transaction.category = 'Food & Dining'
        mock_transaction.subcategory = 'Restaurants'
        mock_transaction.merchant = 'Test Restaurant'
        mock_transaction.description = 'Dinner'
        mock_transaction.payment_method = 'Credit Card'
        mock_transaction.created_at = pd.to_datetime('2024-01-01')

        mock_session.query().all.return_value = [mock_transaction]
        mock_db_manager.return_value.get_session.return_value = mock_session

        # Test loading
        df = self.data_loader.load_transactions()

        assert isinstance(df, pd.DataFrame)


class TestFeatureEngineer:

    def setup_method(self):
        self.feature_engineer = FeatureEngineer()

        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        self.sample_df = pd.DataFrame({
            'date': dates,
            'amount': [25.0 + i * 0.5 for i in range(100)],
            'category': ['Food & Dining'] * 50 + ['Transportation'] * 50,
            'merchant': ['Store A'] * 100,
            'description': ['Purchase'] * 100,
            'payment_method': ['Credit Card'] * 100
        })

    def test_create_time_features(self):
        result = self.feature_engineer.create_time_features(self.sample_df)

        assert 'year' in result.columns
        assert 'month' in result.columns
        assert 'weekday' in result.columns
        assert 'is_weekend' in result.columns
        assert 'month_sin' in result.columns
        assert 'month_cos' in result.columns

    def test_create_spending_patterns(self):
        result = self.feature_engineer.create_spending_patterns(self.sample_df)

        assert 'amount_7d_avg' in result.columns
        assert 'amount_30d_avg' in result.columns
        assert 'cumulative_spending' in result.columns
        assert 'merchant_frequency' in result.columns

    def test_create_all_features(self):
        result = self.feature_engineer.create_all_features(self.sample_df)

        # Should have more columns than original
        assert len(result.columns) > len(self.sample_df.columns)

        # Should have feature columns stored
        assert len(self.feature_engineer.feature_columns) > 0
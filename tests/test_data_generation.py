import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.data_generation.transaction_generator import TransactionGenerator


class TestTransactionGenerator:

    def setup_method(self):
        self.generator = TransactionGenerator(seed=42)

    def test_generator_initialization(self):
        assert self.generator is not None
        assert len(self.generator.categories) == 5
        assert 'Food & Dining' in self.generator.categories

    def test_generate_single_transaction(self):
        test_date = datetime(2024, 1, 15)
        transaction = self.generator._generate_single_transaction(test_date)

        assert isinstance(transaction, dict)
        assert 'date' in transaction
        assert 'amount' in transaction
        assert 'category' in transaction
        assert transaction['amount'] > 0
        assert transaction['date'] == test_date.date()

    def test_generate_transactions_date_range(self):
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)

        transactions = self.generator.generate_transactions(start_date, end_date)

        assert len(transactions) > 0
        assert all(isinstance(t, dict) for t in transactions)

        # Check date range
        dates = [t['date'] for t in transactions]
        assert min(dates) >= start_date.date()
        assert max(dates) <= end_date.date()

    def test_seasonal_variation(self):
        # Test December (holiday season) vs January
        dec_date = datetime(2024, 12, 15)
        jan_date = datetime(2024, 1, 15)

        dec_transactions = [
            self.generator._generate_single_transaction(dec_date)
            for _ in range(100)
        ]
        jan_transactions = [
            self.generator._generate_single_transaction(jan_date)
            for _ in range(100)
        ]

        dec_avg = sum(t['amount'] for t in dec_transactions) / len(dec_transactions)
        jan_avg = sum(t['amount'] for t in jan_transactions) / len(jan_transactions)

        # December should have higher average spending
        assert dec_avg > jan_avg

    def test_category_distribution(self):
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 2, 1)

        transactions = self.generator.generate_transactions(start_date, end_date, avg_transactions_per_day=10)

        categories = [t['category'] for t in transactions]
        category_counts = pd.Series(categories).value_counts()

        # Food & Dining should be most frequent (highest weight)
        assert category_counts.index[0] == 'Food & Dining'
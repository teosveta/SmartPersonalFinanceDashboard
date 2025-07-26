import pandas as pd
from sqlalchemy import text
import sys
import os
import logging

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from src.utils.database import DatabaseManager, Transaction
except ImportError:
    # Fallback import
    import importlib.util

    spec = importlib.util.spec_from_file_location("database",
                                                  os.path.join(os.path.dirname(__file__), '..', 'utils', 'database.py'))
    database_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(database_module)
    DatabaseManager = database_module.DatabaseManager
    Transaction = database_module.Transaction

from typing import Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self):
        self.db_manager = DatabaseManager()

    def load_transactions(self, start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          categories: Optional[List[str]] = None) -> pd.DataFrame:
        """Load transactions from database with optional filters"""

        session = self.db_manager.get_session()
        try:
            query = session.query(Transaction)

            if start_date:
                query = query.filter(Transaction.date >= start_date)
            if end_date:
                query = query.filter(Transaction.date <= end_date)
            if categories:
                query = query.filter(Transaction.category.in_(categories))

            transactions = query.all()

            # Convert to DataFrame
            data = []
            for trans in transactions:
                data.append({
                    'id': trans.id,
                    'date': trans.date,
                    'amount': trans.amount,
                    'category': trans.category,
                    'subcategory': trans.subcategory,
                    'merchant': trans.merchant,
                    'description': trans.description,
                    'payment_method': trans.payment_method,
                    'created_at': trans.created_at
                })

            df = pd.DataFrame(data)
            logger.info(f"Loaded {len(df)} transactions")
            return df

        except Exception as e:
            logger.error(f"Error loading transactions: {e}")
            return pd.DataFrame()
        finally:
            self.db_manager.close_session(session)

    def get_transaction_summary(self) -> dict:
        """Get basic statistics about transactions"""
        session = self.db_manager.get_session()
        try:
            # Get total count
            total_count = session.query(Transaction).count()

            # Get date range
            date_range_query = session.query(
                text("MIN(date) as min_date, MAX(date) as max_date")
            ).first()

            # Get total amount
            total_amount_query = session.query(
                text("SUM(amount) as total_amount")
            ).first()

            # Get category breakdown
            category_query = session.query(
                Transaction.category,
                text("COUNT(*) as count, SUM(amount) as total")
            ).group_by(Transaction.category).all()

            summary = {
                'total_transactions': total_count,
                'date_range': {
                    'start': date_range_query.min_date if date_range_query else None,
                    'end': date_range_query.max_date if date_range_query else None
                },
                'total_amount': float(total_amount_query.total_amount) if total_amount_query.total_amount else 0.0,
                'category_breakdown': [
                    {
                        'category': cat.category,
                        'count': cat.count,
                        'total_amount': float(cat.total)
                    }
                    for cat in category_query
                ]
            }

            return summary

        except Exception as e:
            logger.error(f"Error getting transaction summary: {e}")
            return {}
        finally:
            self.db_manager.close_session(session)

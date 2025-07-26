from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker  # Fixed import
from datetime import datetime
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from src.utils.config import config
except ImportError:
    # Fallback config if import fails
    class FallbackConfig:
        def __init__(self):
            self.project_root = Path(__file__).parent.parent.parent
            self.data_dir = self.project_root / "data"
            self.data_dir.mkdir(parents=True, exist_ok=True)

        @property
        def database_url(self):
            return f"sqlite:///{self.data_dir}/finance_dashboard.db"


    config = FallbackConfig()

# Use the modern declarative_base import
Base = declarative_base()


class Transaction(Base):
    __tablename__ = 'transactions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    amount = Column(Float, nullable=False)
    category = Column(String(50), nullable=False)
    subcategory = Column(String(50))
    merchant = Column(String(100))
    description = Column(Text)
    payment_method = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Transaction(id={self.id}, date={self.date}, amount={self.amount}, category='{self.category}')>"


class Budget(Base):
    __tablename__ = 'budgets'

    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String(50), nullable=False)
    monthly_limit = Column(Float, nullable=False)
    year = Column(Integer, nullable=False)
    month = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Budget(category='{self.category}', limit={self.monthly_limit}, month={self.month}/{self.year})>"


class Prediction(Base):
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_date = Column(Date, nullable=False)
    category = Column(String(50), nullable=False)
    predicted_amount = Column(Float, nullable=False)
    confidence_score = Column(Float)
    model_version = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Prediction(date={self.prediction_date}, category='{self.category}', amount={self.predicted_amount})>"


class DatabaseManager:
    def __init__(self):
        try:
            self.engine = create_engine(config.database_url, echo=False)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            self.create_tables()
            print(f"✅ Database initialized successfully at: {config.database_url}")
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            raise

    def create_tables(self):
        """Create all tables in the database"""
        try:
            Base.metadata.create_all(bind=self.engine)
            print("✅ Database tables created successfully")
        except Exception as e:
            print(f"❌ Error creating tables: {e}")
            raise

    def get_session(self):
        """Get a new database session"""
        return self.SessionLocal()

    def close_session(self, session):
        """Close a database session"""
        if session:
            session.close()

    def test_connection(self):
        """Test database connection and basic operations"""
        try:
            session = self.get_session()

            # Test basic query
            result = session.execute("SELECT 1 as test").fetchone()

            # Test transaction count
            count = session.query(Transaction).count()

            session.close()

            return {
                'connection': True,
                'test_query': result[0] == 1 if result else False,
                'transaction_count': count
            }
        except Exception as e:
            return {
                'connection': False,
                'error': str(e),
                'transaction_count': 0
            }

    def get_database_info(self):
        """Get database information and statistics"""
        try:
            session = self.get_session()

            # Table counts
            transaction_count = session.query(Transaction).count()
            budget_count = session.query(Budget).count()
            prediction_count = session.query(Prediction).count()

            # Date range if transactions exist
            date_info = {}
            if transaction_count > 0:
                earliest = session.query(Transaction.date).order_by(Transaction.date.asc()).first()
                latest = session.query(Transaction.date).order_by(Transaction.date.desc()).first()
                if earliest and latest:
                    date_info = {
                        'earliest_transaction': earliest[0],
                        'latest_transaction': latest[0]
                    }

            # Category breakdown
            categories = {}
            if transaction_count > 0:
                category_data = session.query(
                    Transaction.category,
                    session.query().func.count(Transaction.id),
                    session.query().func.sum(Transaction.amount)
                ).group_by(Transaction.category).all()

                categories = {
                    cat: {'count': count, 'total': float(total)}
                    for cat, count, total in category_data
                }

            session.close()

            return {
                'tables': {
                    'transactions': transaction_count,
                    'budgets': budget_count,
                    'predictions': prediction_count
                },
                'date_range': date_info,
                'categories': categories,
                'database_path': str(config.database_url)
            }

        except Exception as e:
            return {
                'error': str(e),
                'database_path': str(config.database_url)
            }

"""
Simple script to test database functionality
Run with: python test_database.py
"""


def test_database_setup():
    """Test database setup and functionality"""
    print("🔧 Testing Database Setup...")
    print("=" * 50)

    try:
        # Import and initialize
        from src.utils.database import DatabaseManager, Transaction
        from src.data_generation.transaction_generator import TransactionGenerator
        from datetime import datetime, timedelta

        print("✅ Successfully imported database modules")

        # Initialize database
        db_manager = DatabaseManager()
        print("✅ Database manager initialized")

        # Test connection
        connection_test = db_manager.test_connection()
        print(f"✅ Connection test: {connection_test}")

        # Get database info
        db_info = db_manager.get_database_info()
        print(f"📊 Database info: {db_info}")

        # Check if we need to generate data
        if db_info.get('tables', {}).get('transactions', 0) == 0:
            print("\n🎲 No data found. Generating sample data...")

            generator = TransactionGenerator()
            transactions = generator.generate_transactions(
                start_date=datetime.now() - timedelta(days=90),  # 3 months
                end_date=datetime.now(),
                avg_transactions_per_day=3
            )

            generator.save_to_database(transactions, db_manager)
            print(f"✅ Generated {len(transactions)} sample transactions")

            # Get updated info
            db_info = db_manager.get_database_info()
            print(f"📊 Updated database info: {db_info}")

        print("\n🎉 Database test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_database_setup()
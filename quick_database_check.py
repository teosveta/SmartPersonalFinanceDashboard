"""
Quick database status checker
"""


def quick_check():
    """Quick database status check"""
    try:
        # Simple import test
        import sqlite3
        import os
        from pathlib import Path

        # Check if database file exists
        project_root = Path(__file__).parent.parent.parent
        data_dir = project_root / "data"
        db_path = data_dir / "finance_dashboard.db"

        print(f"🔍 Looking for database at: {db_path}")

        if db_path.exists():
            print("✅ Database file exists")

            # Connect and check tables
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Check tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print(f"📋 Tables found: {[table[0] for table in tables]}")

            # Check transaction count
            if ('transactions',) in tables:
                cursor.execute("SELECT COUNT(*) FROM transactions;")
                count = cursor.fetchone()[0]
                print(f"📊 Transaction count: {count:,}")

                if count > 0:
                    cursor.execute("SELECT MIN(date), MAX(date) FROM transactions;")
                    date_range = cursor.fetchone()
                    print(f"📅 Date range: {date_range[0]} to {date_range[1]}")

            conn.close()
            print("✅ Database check completed successfully!")

        else:
            print("⚠️ Database file not found")
            print("💡 Run the dashboard to create it: streamlit run enhanced_run.py")

    except Exception as e:
        print(f"❌ Database check failed: {e}")


if __name__ == "__main__":
    quick_check()

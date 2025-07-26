"""
Simple launcher that works if enhanced version has issues
"""
import streamlit as st
import sys
import os

# Add path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

st.set_page_config(
    page_title="Smart Finance Dashboard",
    page_icon="💰",
    layout="wide"
)


def simple_main():
    st.title("💰 Smart Personal Finance Dashboard")

    # Try enhanced version first
    try:
        from src.dashboard.enhanced_app import EnhancedFinanceDashboard
        dashboard = EnhancedFinanceDashboard()
        dashboard.run()

    except Exception as enhanced_error:
        st.warning(f"Enhanced features unavailable: {enhanced_error}")

        # Fallback to basic version
        try:
            from src.dashboard.app import FinanceDashboard
            st.info("🔄 Falling back to basic version...")
            basic_dashboard = FinanceDashboard()
            basic_dashboard.run()

        except Exception as basic_error:
            st.error("❌ Both enhanced and basic versions failed to load")
            st.error(f"Enhanced error: {enhanced_error}")
            st.error(f"Basic error: {basic_error}")

            # Show manual instructions
            st.markdown("### 🛠️ Manual Setup Required")
            st.markdown("""
            **Please ensure you have:**
            1. All required Python packages: `pip install -r requirements.txt`
            2. Proper file structure with all source files
            3. Database initialized (run basic version first)

            **Try running:**
            ```bash
            # Basic version
            streamlit run run.py

            # Generate sample data first
            python -c "
            from src.data_generation.transaction_generator import TransactionGenerator
            from src.utils.database import DatabaseManager
            generator = TransactionGenerator()
            db_manager = DatabaseManager()
            generator.generate_and_save(months_back=12, db_manager=db_manager)
            print('Sample data generated!')
            "
            ```
            """)


simple_main()
import streamlit as st
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from src.dashboard.app import FinanceDashboard
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all required files are in place and try again.")
    st.stop()

if __name__ == "__main__":
    try:
        dashboard = FinanceDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.error("Please check the logs and try again.")
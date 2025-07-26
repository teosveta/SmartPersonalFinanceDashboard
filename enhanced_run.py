import streamlit as st
import sys
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Smart Personal Finance Dashboard Pro",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/finance-dashboard',
        'Report a bug': "https://github.com/yourusername/finance-dashboard/issues",
        'About': "# Smart Personal Finance Dashboard Pro\n"
                 "Advanced AI-powered financial analytics platform\n"
                 "Built with Streamlit, scikit-learn, and XGBoost"
    }
)


def main():
    """Main application entry point"""
    try:
        # Import the enhanced dashboard
        from src.dashboard.enhanced_app import EnhancedFinanceDashboard

        # Initialize and run the dashboard
        logger.info("Starting Enhanced Finance Dashboard")
        dashboard = EnhancedFinanceDashboard()
        dashboard.run()

    except ImportError as e:
        st.error("❌ Import Error")
        st.error(f"Could not import required modules: {e}")
        st.error("Please ensure all source files are properly installed.")

        with st.expander("🔧 Troubleshooting Steps", expanded=True):
            st.markdown("""
            **Common solutions:**
            1. **Install dependencies**: `pip install -r requirements.txt`
            2. **Check file structure**: Ensure all files are in correct directories
            3. **Run setup**: `python setup_project.py`
            4. **Try basic version**: `streamlit run run.py`

            **Required files for enhanced version:**
            ```
            src/dashboard/enhanced_app.py
            src/models/advanced_predictor.py
            src/analytics/advanced_insights.py
            src/dashboard/components/advanced_charts.py
            src/dashboard/components/interactive_features.py
            ```

            **Missing files? Create them from the artifacts or run basic version:**
            ```bash
            streamlit run run.py
            ```
            """)

        st.stop()

    except Exception as e:
        st.error("❌ Application Error")
        st.error(f"An unexpected error occurred: {e}")

        with st.expander("🐛 Error Details", expanded=False):
            import traceback
            st.code(traceback.format_exc(), language="python")

        st.info("💡 Try refreshing the page or running the basic version: `streamlit run run.py`")

        logger.error(f"Application error: {e}", exc_info=True)


# Run the main function directly (no if __name__ == "__main__" needed for streamlit)
main()
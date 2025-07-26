"""
Tool to check if all components are properly installed
"""
import streamlit as st
import sys
import os
from pathlib import Path


def check_project_setup():
    """Check if project is properly set up"""
    st.title("🔧 Project Setup Checker")

    checks = []

    # Check directory structure
    required_dirs = [
        'src',
        'src/utils',
        'src/etl',
        'src/models',
        'src/dashboard',
        'src/dashboard/components',
        'src/analytics',
        'src/data_generation'
    ]

    st.subheader("📁 Directory Structure")
    for directory in required_dirs:
        exists = Path(directory).exists()
        checks.append(exists)
        status = "✅" if exists else "❌"
        st.write(f"{status} {directory}")

    # Check required files
    required_files = [
        'src/utils/database.py',
        'src/etl/data_loader.py',
        'src/models/expense_predictor.py',
        'src/dashboard/app.py',
        'requirements.txt'
    ]

    st.subheader("📄 Core Files")
    for file_path in required_files:
        exists = Path(file_path).exists()
        checks.append(exists)
        status = "✅" if exists else "❌"
        st.write(f"{status} {file_path}")

    # Check enhanced files
    enhanced_files = [
        'src/dashboard/enhanced_app.py',
        'src/models/advanced_predictor.py',
        'src/analytics/advanced_insights.py',
        'src/dashboard/components/advanced_charts.py',
        'src/dashboard/components/interactive_features.py'
    ]

    st.subheader("🚀 Enhanced Files")
    enhanced_count = 0
    for file_path in enhanced_files:
        exists = Path(file_path).exists()
        if exists:
            enhanced_count += 1
        status = "✅" if exists else "❌"
        st.write(f"{status} {file_path}")

    # Check Python packages
    st.subheader("📦 Python Packages")
    required_packages = [
        'pandas', 'numpy', 'streamlit', 'plotly', 'scikit-learn',
        'sqlalchemy', 'faker', 'xgboost', 'scipy'
    ]

    package_count = 0
    for package in required_packages:
        try:
            __import__(package)
            package_count += 1
            st.write(f"✅ {package}")
        except ImportError:
            st.write(f"❌ {package} - Run: pip install {package}")

    # Summary
    st.subheader("📊 Setup Summary")

    basic_ready = sum(checks) >= len(required_dirs) + len(required_files)
    enhanced_ready = enhanced_count >= len(enhanced_files)
    packages_ready = package_count >= len(required_packages)

    col1, col2, col3 = st.columns(3)

    with col1:
        color1 = "green" if basic_ready else "red"
        st.markdown(
            f"**Basic Version:** <span style='color:{color1}'>{'✅ Ready' if basic_ready else '❌ Issues'}</span>",
            unsafe_allow_html=True)

    with col2:
        color2 = "green" if enhanced_ready else "orange"
        st.markdown(
            f"**Enhanced Version:** <span style='color:{color2}'>{'✅ Ready' if enhanced_ready else '⚠️ Missing Files'}</span>",
            unsafe_allow_html=True)

    with col3:
        color3 = "green" if packages_ready else "red"
        st.markdown(
            f"**Dependencies:** <span style='color:{color3}'>{'✅ Installed' if packages_ready else '❌ Missing'}</span>",
            unsafe_allow_html=True)

    # Recommendations
    st.subheader("💡 Recommendations")

    if not packages_ready:
        st.error("🚨 **Priority**: Install missing Python packages")
        st.code("pip install -r requirements.txt")

    if not basic_ready:
        st.warning("⚠️ **Setup Required**: Create missing directories and core files")
        st.code("python setup_project.py")

    if basic_ready and not enhanced_ready:
        st.info("ℹ️ **Optional**: Add enhanced files for advanced features")
        st.write("You can run the basic version with: `streamlit run run.py`")

    if basic_ready and enhanced_ready and packages_ready:
        st.success("🎉 **All Set!** Your project is fully configured")
        st.write("Run the enhanced version with: `streamlit run enhanced_run.py`")
        st.write("Or run the basic version with: `streamlit run run.py`")


# Run checker if this file is executed directly
if __name__ == "__main__":
    check_project_setup()
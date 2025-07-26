from setuptools import setup, find_packages

setup(
    name="finance_dashboard",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.3",
        "numpy>=1.24.3",
        "scikit-learn>=1.3.0",
        "plotly>=5.15.0",
        "streamlit>=1.25.0",
        "sqlalchemy>=2.0.19",
        "statsmodels>=0.14.0",
        "faker>=19.3.1",
        "python-dateutil>=2.8.2",
        "pyyaml>=6.0.1",
        "pytest>=7.4.0"
    ],
    author="AI/ML Data Engineer",
    description="Smart Personal Finance Dashboard with Predictive Analytics",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
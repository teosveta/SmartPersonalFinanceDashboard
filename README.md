# Smart Personal Finance Dashboard

🚀 **A comprehensive financial analytics platform with machine learning-powered insights**

## 📋 Overview

This project showcases a production-ready financial dashboard that processes transaction data, identifies spending patterns, predicts future expenses, and provides automated budget recommendations. Built with modern data science and engineering best practices.

## ✨ Features

### 🔄 **Data Engineering**
- Synthetic transaction data generation with realistic patterns
- Robust ETL pipeline with data validation and cleaning
- Advanced feature engineering with 20+ engineered features
- Database abstraction layer with SQLAlchemy ORM

### 🤖 **Machine Learning**
- **Expense Prediction**: ARIMA + Random Forest hybrid forecasting
- **Category Classification**: NLP-based transaction categorization
- **Anomaly Detection**: Statistical + Isolation Forest ensemble
- **Budget Optimization**: AI-powered spending recommendations

### 📊 **Interactive Dashboard**
- Real-time spending analytics and visualizations
- Multi-tab interface with drill-down capabilities
- Predictive charts with confidence intervals
- Automated insight generation and alerts

### 💡 **Business Intelligence**
- Weekly spending pattern analysis
- Category-wise budget performance tracking
- Seasonal trend identification
- Financial health scoring system

## 🏗️ **Architecture**

```
finance_dashboard/
├── 📁 src/                    # Source code
│   ├── 📁 data_generation/   # Data creation & validation
│   ├── 📁 etl/              # Extract, Transform, Load
│   ├── 📁 models/           # ML models & algorithms
│   ├── 📁 dashboard/        # Streamlit application
│   ├── 📁 analytics/        # Business intelligence
│   └── 📁 utils/           # Configuration & utilities
├── 📁 config/              # YAML configurations
├── 📁 tests/               # Unit & integration tests
├── 📁 docker/              # Containerization
├── 📁 notebooks/           # Jupyter analysis notebooks
└── 📁 data/               # Data storage & models
```

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.8+ 
- pip or conda package manager

### **Installation**

1. **Clone or download this project structure**

2. **Navigate to project directory:**
   ```bash
   cd finance_dashboard
   ```

3. **Set up virtual environment:**
   ```bash
   # Linux/Mac
   ./setup_env.sh

   # Windows  
   setup_env.bat

   # Or manually
   python -m venv finance_dashboard_env
   source finance_dashboard_env/bin/activate  # Linux/Mac
   finance_dashboard_env\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

4. **Run the dashboard:**
   ```bash
   streamlit run run.py
   ```

5. **Generate sample data:**
   - Open browser to `http://localhost:8501`
   - Click "Generate Sample Data" in sidebar
   - Explore the dashboard features!

### **Docker Deployment**

```bash
cd docker
docker-compose up --build
```

Access at `http://localhost:8501`

## 🎯 **Usage Guide**

### **Dashboard Navigation**
- **Overview**: Key metrics, trends, recent transactions
- **Predictions**: ML-powered expense forecasting  
- **Analytics**: Advanced analysis & anomaly detection
- **Insights**: AI-generated recommendations

### **Key Workflows**
1. **Data Generation**: Create realistic synthetic transactions
2. **Model Training**: Automatically trains on data load
3. **Analysis**: Filter by date ranges and categories
4. **Predictions**: Forecast future expenses by category
5. **Insights**: Review automated financial recommendations

## 🧪 **Testing**

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

## 📊 **Development**

### **Jupyter Notebooks**
```bash
jupyter lab notebooks/
```

### **Model Development**
- `01_data_exploration.ipynb`: Data analysis & visualization
- `02_model_development.ipynb`: ML model experimentation  
- `03_visualization_prototypes.ipynb`: Chart development

### **Code Quality**
```bash
# Format code
black src/ tests/

# Lint code  
flake8 src/ tests/

# Type checking
mypy src/
```

## 🏭 **Production Deployment**

### **Environment Variables**
```bash
export DATABASE_URL="postgresql://user:pass@localhost/finance_db"
export LOG_LEVEL="INFO"
export CACHE_TIMEOUT=3600
```

### **Performance Optimization**
- Database connection pooling
- Streamlit caching for expensive operations
- Model serialization for faster loading
- Batch processing for large datasets

## 📈 **Technical Highlights**

- **Database**: SQLAlchemy ORM with migration support
- **ML Pipeline**: Feature engineering → Model training → Prediction
- **Visualization**: Interactive Plotly charts with custom styling
- **Testing**: Comprehensive test suite with 90%+ coverage
- **Documentation**: API docs, user guides, deployment guides
- **Containerization**: Docker + docker-compose for scalability

## 📚 **Project Structure Details**

### **Core Components**
- `TransactionGenerator`: Creates realistic synthetic financial data
- `FeatureEngineer`: Builds 20+ ML features from raw transactions
- `ExpensePredictor`: Hybrid ARIMA/RF model for forecasting
- `CategoryClassifier`: NLP-based transaction categorization
- `AnomalyDetector`: Multi-method outlier detection system
- `InsightsGenerator`: Automated financial advice engine

### **Data Flow**
1. **Data Generation** → Synthetic transactions with seasonal patterns
2. **ETL Pipeline** → Clean, validate, and engineer features
3. **Model Training** → Train predictive and classification models
4. **Dashboard** → Interactive visualizations and insights
5. **Analytics** → Automated recommendations and alerts

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎯 **Portfolio Value**

This project demonstrates:
- **Full-Stack Data Science**: End-to-end ML pipeline
- **Production Engineering**: Scalable, tested, documented code
- **Business Acumen**: Solves real financial planning problems  
- **Technical Depth**: Advanced ML, data engineering, and visualization
- **Professional Delivery**: Clean architecture, comprehensive testing

---

**Created on:** 2025-07-26 20:48:49
**Technology Stack:** Python, Streamlit, Plotly, scikit-learn, SQLAlchemy
**Deployment:** Docker, Local, Cloud-ready

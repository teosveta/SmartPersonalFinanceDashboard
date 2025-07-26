import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re
import joblib
import logging
from typing import List, Dict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CategoryClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False

        # Define merchant patterns for rule-based classification
        self.merchant_patterns = {
            'Food & Dining': [
                r'(restaurant|cafe|coffee|pizza|burger|food|dining|kitchen|grill|bistro)',
                r'(starbucks|mcdonalds|subway|chipotle|dominos|kfc)',
                r'(grocery|market|walmart|target|kroger|safeway)'
            ],
            'Transportation': [
                r'(gas|fuel|shell|exxon|bp|chevron|mobil)',
                r'(uber|lyft|taxi|metro|transit|parking)',
                r'(auto|car|tire|oil|mechanic)'
            ],
            'Shopping': [
                r'(amazon|ebay|store|shop|mall|retail)',
                r'(clothing|apparel|fashion|shoes)',
                r'(electronics|best buy|apple|samsung)'
            ],
            'Entertainment': [
                r'(movie|cinema|theater|netflix|spotify|gaming)',
                r'(concert|event|ticket|entertainment)',
                r'(gym|fitness|sports)'
            ],
            'Bills & Utilities': [
                r'(electric|gas|water|utility|bill|payment)',
                r'(insurance|phone|internet|cable)',
                r'(bank|fee|interest|loan)'
            ]
        }

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for classification"""
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def rule_based_classify(self, merchant: str, description: str) -> str:
        """Apply rule-based classification using merchant patterns"""
        text = f"{merchant} {description}".lower()

        for category, patterns in self.merchant_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return category

        return None

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare text features for classification"""
        # Combine merchant and description
        df['combined_text'] = (
                df['merchant'].fillna('') + ' ' +
                df['description'].fillna('')
        ).apply(self.preprocess_text)

        return df['combined_text']

    def train(self, df: pd.DataFrame) -> Dict:
        """Train the category classifier"""
        # Prepare features
        text_features = self.prepare_features(df)

        # Vectorize text
        X = self.vectorizer.fit_transform(text_features)
        y = df['category']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train classifier
        self.classifier.fit(X_train, y_train)

        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        self.is_trained = True

        logger.info(f"Category classifier trained - Accuracy: {accuracy:.3f}")

        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

    def predict(self, merchant: str, description: str) -> Dict:
        """Predict category for a single transaction"""
        # Try rule-based first
        rule_prediction = self.rule_based_classify(merchant, description)
        if rule_prediction:
            return {
                'category': rule_prediction,
                'confidence': 0.95,
                'method': 'rule_based'
            }

        # Fall back to ML model
        if not self.is_trained:
            return {
                'category': 'Unknown',
                'confidence': 0.0,
                'method': 'none'
            }

        # Prepare text
        text = f"{merchant} {description}"
        processed_text = self.preprocess_text(text)

        # Vectorize and predict
        text_vector = self.vectorizer.transform([processed_text])
        prediction = self.classifier.predict(text_vector)[0]
        probabilities = self.classifier.predict_proba(text_vector)[0]
        confidence = max(probabilities)

        return {
            'category': prediction,
            'confidence': confidence,
            'method': 'ml_model'
        }

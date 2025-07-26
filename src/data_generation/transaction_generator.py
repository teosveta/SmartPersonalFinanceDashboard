import random
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker
import numpy as np
from typing import List, Dict


class TransactionGenerator:
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.fake = Faker()
        Faker.seed(seed)

        self.categories = {
            'Food & Dining': {
                'subcategories': ['Restaurants', 'Groceries', 'Coffee Shops', 'Fast Food'],
                'merchants': {
                    'Restaurants': ['Olive Garden', 'Red Lobster', 'Local Bistro', 'Downtown Grill'],
                    'Groceries': ['Walmart', 'Target', 'Whole Foods', 'Kroger', 'Safeway'],
                    'Coffee Shops': ['Starbucks', 'Dunkin Donuts', 'Local Coffee Co', 'Peets Coffee'],
                    'Fast Food': ['McDonalds', 'Burger King', 'Subway', 'Chipotle', 'Taco Bell']
                },
                'amount_range': (5, 150),
                'frequency_weight': 0.3
            },
            'Transportation': {
                'subcategories': ['Gas', 'Public Transport', 'Parking', 'Car Maintenance'],
                'merchants': {
                    'Gas': ['Shell', 'Exxon', 'BP', 'Chevron', '76 Station'],
                    'Public Transport': ['Metro Transit', 'City Bus', 'Uber', 'Lyft'],
                    'Parking': ['City Parking', 'Airport Parking', 'Mall Parking'],
                    'Car Maintenance': ['Auto Zone', 'Jiffy Lube', 'Local Mechanic', 'Car Wash Plus']
                },
                'amount_range': (10, 200),
                'frequency_weight': 0.25
            },
            'Shopping': {
                'subcategories': ['Clothing', 'Electronics', 'Home & Garden', 'Personal Care'],
                'merchants': {
                    'Clothing': ['Amazon', 'Target', 'Macys', 'H&M', 'Zara', 'Old Navy'],
                    'Electronics': ['Best Buy', 'Amazon', 'Apple Store', 'Micro Center'],
                    'Home & Garden': ['Home Depot', 'Lowes', 'IKEA', 'Bed Bath Beyond'],
                    'Personal Care': ['CVS', 'Walgreens', 'Sephora', 'Ulta Beauty']
                },
                'amount_range': (15, 500),
                'frequency_weight': 0.2
            },
            'Entertainment': {
                'subcategories': ['Movies', 'Sports Events', 'Streaming', 'Gaming'],
                'merchants': {
                    'Movies': ['AMC Theaters', 'Regal Cinemas', 'Local Theater'],
                    'Sports Events': ['Stadium Box Office', 'Ticketmaster', 'StubHub'],
                    'Streaming': ['Netflix', 'Spotify', 'Disney Plus', 'HBO Max'],
                    'Gaming': ['Steam', 'PlayStation Store', 'Xbox Live', 'GameStop']
                },
                'amount_range': (8, 300),
                'frequency_weight': 0.15
            },
            'Bills & Utilities': {
                'subcategories': ['Electricity', 'Internet', 'Phone', 'Insurance'],
                'merchants': {
                    'Electricity': ['Pacific Gas Electric', 'Con Edison', 'Duke Energy'],
                    'Internet': ['Comcast', 'Verizon', 'AT&T', 'Spectrum'],
                    'Phone': ['Verizon Wireless', 'AT&T Mobility', 'T-Mobile'],
                    'Insurance': ['State Farm', 'Geico', 'Allstate', 'Progressive']
                },
                'amount_range': (50, 300),
                'frequency_weight': 0.1
            }
        }

        self.payment_methods = ['Credit Card', 'Debit Card', 'Cash', 'Digital Wallet']

    def generate_transactions(self, start_date: datetime, end_date: datetime,
                              avg_transactions_per_day: int = 3) -> List[Dict]:
        transactions = []
        current_date = start_date

        while current_date <= end_date:
            # Vary transaction count based on day of week (more on weekends)
            day_multiplier = 1.3 if current_date.weekday() >= 5 else 1.0
            daily_transactions = np.random.poisson(avg_transactions_per_day * day_multiplier)

            for _ in range(daily_transactions):
                transaction = self._generate_single_transaction(current_date)
                transactions.append(transaction)

            current_date += timedelta(days=1)

        return transactions

    def _generate_single_transaction(self, date: datetime) -> Dict:
        # Select category based on weights
        categories = list(self.categories.keys())
        weights = [self.categories[cat]['frequency_weight'] for cat in categories]
        category = np.random.choice(categories, p=np.array(weights) / sum(weights))

        # Select subcategory
        subcategories = self.categories[category]['subcategories']
        subcategory = random.choice(subcategories)

        # Select merchant
        merchants = self.categories[category]['merchants'][subcategory]
        merchant = random.choice(merchants)

        # Generate amount with some seasonality and randomness
        min_amount, max_amount = self.categories[category]['amount_range']

        # Add seasonal variation (higher spending in December, lower in January)
        seasonal_multiplier = 1.0
        if date.month == 12:  # December - holiday spending
            seasonal_multiplier = 1.4
        elif date.month == 1:  # January - post-holiday tightening
            seasonal_multiplier = 0.7
        elif date.month in [6, 7, 8]:  # Summer months
            seasonal_multiplier = 1.1

        base_amount = random.uniform(min_amount, max_amount)
        amount = round(base_amount * seasonal_multiplier, 2)

        # Generate description
        description = self._generate_description(category, subcategory, merchant)

        return {
            'date': date.date(),
            'amount': amount,
            'category': category,
            'subcategory': subcategory,
            'merchant': merchant,
            'description': description,
            'payment_method': random.choice(self.payment_methods)
        }

    def _generate_description(self, category: str, subcategory: str, merchant: str) -> str:
        descriptions = {
            'Food & Dining': [
                f"Dinner at {merchant}",
                f"Weekly grocery shopping",
                f"Quick lunch",
                f"Coffee and pastry",
                f"Weekend meal"
            ],
            'Transportation': [
                f"Gas fill-up",
                f"Monthly transit pass",
                f"Parking downtown",
                f"Oil change service",
                f"Car wash"
            ],
            'Shopping': [
                f"Online purchase from {merchant}",
                f"Weekend shopping",
                f"New clothes",
                f"Home improvement supplies",
                f"Personal care items"
            ],
            'Entertainment': [
                f"Movie tickets",
                f"Monthly subscription",
                f"Gaming purchase",
                f"Concert tickets",
                f"Streaming service"
            ],
            'Bills & Utilities': [
                f"Monthly {subcategory.lower()} bill",
                f"Utility payment",
                f"Insurance premium",
                f"Service fee"
            ]
        }

        return random.choice(descriptions.get(category, [f"Purchase from {merchant}"]))

    def save_to_database(self, transactions: List[Dict], db_manager):
        session = db_manager.get_session()
        try:
            from ..utils.database import Transaction

            for trans_data in transactions:
                transaction = Transaction(**trans_data)
                session.add(transaction)

            session.commit()
            print(f"Successfully saved {len(transactions)} transactions to database")

        except Exception as e:
            session.rollback()
            print(f"Error saving transactions: {e}")
        finally:
            db_manager.close_session(session)

    def generate_and_save(self, months_back: int = 12, db_manager=None):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months_back * 30)

        print(f"Generating transactions from {start_date.date()} to {end_date.date()}")
        transactions = self.generate_transactions(start_date, end_date)

        if db_manager:
            self.save_to_database(transactions, db_manager)

        return transactions

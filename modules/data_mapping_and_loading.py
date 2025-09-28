import pandas as pd
from faker import Faker
import random
from sqlalchemy import create_engine, text
import sqlalchemy
import psycopg2
import numpy as np

# ----------------------------
# Load real transaction data
# ----------------------------
df_transactions = pd.read_csv('./downloads/creditcard.csv')

# ----------------------------
# Initialize Faker and seeds
# ----------------------------
fake = Faker()
Faker.seed(42)  
random.seed(42)

# ----------------------------
# Generate simulated customers
# ----------------------------
num_customers = 1000  # adjustable for number of customers

customers = []
age_groups = ["18-25", "26-35", "36-45", "46-60", "60+"]
countries = ["China", "USA", "UK", "India", "Australia", "Argentina"]

for i in range(1, num_customers + 1):
    customer = {
        "customer_id": i,
        "gender": random.choice(["Male", "Female"]),
        "age_group": random.choice(age_groups),
        "country": random.choice(countries)
    }
    customers.append(customer)

df_customers = pd.DataFrame(customers)
# print(df_customers.head())  # debug print

# ----------------------------
# Generate simulated cards
# ----------------------------
num_cards = 500  # adjustable for number of cards

card_types = ["Credit", "Debit"]
banks = ["ICBC", "Bank of China", "HSBC", "Citi", "Chase"]

cards = []
for i in range(1, num_cards + 1):
    card = {
        "card_id": i,
        "card_type": random.choice(card_types),
        "bank": random.choice(banks)
    }
    cards.append(card)

df_cards = pd.DataFrame(cards)
# print(df_cards.head())  # debug print

# ----------------------------
# Generate simulated merchants
# ----------------------------
num_merchants = 200  # adjustable for number of merchants

categories = ["Grocery", "Electronics", "Travel", "Dining", "Clothing"]
regions = ["China", "USA", "UK", "India", "Australia", "Argentina"]

merchants = []
for i in range(1, num_merchants + 1):
    merchant = {
        "merchant_id": i,
        "category": random.choice(categories),
        "region": random.choice(regions)
    }
    merchants.append(merchant)

df_merchants = pd.DataFrame(merchants)
# print(df_merchants.head())  # debug print

# ----------------------------
# Connect to Neon database
# ----------------------------
connection_string = "postgresql://neondb_owner:npg_NJo63LxhiQFE@ep-square-unit-ac9wcwpb-pooler.sa-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
engine = create_engine(connection_string)

# Test connection
with engine.connect() as conn:
    result = conn.execute(text("SELECT 1;"))
    print(result.fetchone())

# ----------------------------
# Link cards to customers
# ----------------------------
available_cards = df_cards['card_id'].tolist()  # list all available cards
card_owner_mapping = {}  # dict for mapping cards -> owners

for customer_id in df_customers['customer_id']:
    # choose between 1 to 3 cards for each customer, but never more than available
    num_cards_per_customer = min(np.random.randint(1, 4), len(available_cards))
    
    if num_cards_per_customer == 0:
        continue  # no cards left to assign
    
    # pick cards from the remaining pool
    assigned = np.random.choice(available_cards, size=num_cards_per_customer, replace=False)
    
    for card_id in assigned:
        card_owner_mapping[card_id] = customer_id  # map card to customer
        available_cards.remove(card_id)  # remove card from available pool

# apply mapping to cards dataframe
df_cards['customer_id'] = df_cards['card_id'].map(card_owner_mapping)

# ----------------------------
# Map transactions to customers, cards, and merchants
# ----------------------------
transactions = []

for idx, row in df_transactions.iterrows():
    # choose a random customer
    customer = df_customers.sample(1).iloc[0]
    customer_id = customer['customer_id']
    customer_country = customer['country']

    # pick a card belonging to this customer
    customer_cards = df_cards[df_cards['customer_id'] == customer_id]
    if customer_cards.empty:
        # fallback: pick any card if customer has no assigned card
        card_id = df_cards.sample(1).iloc[0]['card_id']
    else:
        card_id = customer_cards.sample(1).iloc[0]['card_id']

    # determine merchant region
    if np.random.rand() < 0.1:  # ~10% international transactions
        # pick a merchant in a different country
        possible_merchants = df_merchants[df_merchants['region'] != customer_country]
    else:
        # pick a merchant in the same country
        possible_merchants = df_merchants[df_merchants['region'] == customer_country]

    if not possible_merchants.empty:
        merchant = possible_merchants.sample(1).iloc[0]
    else:
        # fallback if no merchants match criteria
        merchant = df_merchants.sample(1).iloc[0]
    merchant_id = merchant['merchant_id']

    transactions.append({
        'transaction_id': idx + 1,
        'customer_id': customer_id,
        'card_id': card_id,
        'merchant_id': merchant_id,
        'amount': row['Amount'],
        'class': row['Class'],
        **{f'V{i}': row[f'V{i}'] for i in range(1, 29)}  # adds V1-V28
    })


# convert to DataFrame
df_transaction_mapping = pd.DataFrame(transactions)

# ----------------------------
# Send all tables to Neon
# ----------------------------
df_customers.to_sql('customers', engine, if_exists='replace', index=False)
df_cards.to_sql('cards', engine, if_exists='replace', index=False)
df_merchants.to_sql('merchants', engine, if_exists='replace', index=False)
df_transaction_mapping.to_sql('transactions', engine, if_exists='replace', index=False, chunksize=5000)

print("All data successfully sent to Neon!")

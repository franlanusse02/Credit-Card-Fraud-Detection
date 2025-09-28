# main.py
# Fraud detection pipeline using mapped transactions
# everything before was in exploration.py, data_mapping_and_loading.py, setup.py

import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Connect to Neon database
# ----------------------------
connection_string = "postgresql://neondb_owner:npg_NJo63LxhiQFE@ep-square-unit-ac9wcwpb-pooler.sa-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
engine = create_engine(connection_string)

# ----------------------------
# Pull transactions with customer and merchant info
# ----------------------------
query = """
SELECT t.*,
       c.country AS customer_country,
       m.region AS merchant_region
FROM transactions t
JOIN customers c ON t.customer_id = c.customer_id
JOIN merchants m ON t.merchant_id = m.merchant_id
"""
df = pd.read_sql(query, engine)

# ----------------------------
# Quick inspection
# ----------------------------
print(df.head())
print(df.isnull().sum())  # check for missing values

# ----------------------------
# Feature engineering
# ----------------------------
# binary flag for international transactions
df['is_international'] = (df['customer_country'] != df['merchant_region']).astype(int)

# ----------------------------
# Scale numeric features
# ----------------------------
numeric_features = ['amount']  # add other numeric columns if needed
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# ----------------------------
# Split data into train/test
# ----------------------------
# X: all features except target and transaction_id
# y: target column 'class'
X = df.drop(columns=['class', 'transaction_id', 'customer_country', 'merchant_region'])
y = df['class']

# split while preserving class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# ----------------------------
# Fit baseline Logistic Regression
# ----------------------------
# using class_weight='balanced' to account for fraud imbalance
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# ----------------------------
# Make predictions
# ----------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ----------------------------
# Evaluate the model
# ----------------------------
print("Classification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
print("ROC AUC:", roc_auc)

# ----------------------------
# Visualize confusion matrix
# ----------------------------
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ==============================
# Coffee Sales Analysis Project
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------
# 1. Load Dataset (CSV)
# ------------------------------
df = pd.read_csv("C:/Users/viswe/OneDrive/Desktop/Coffee_sales.csv")

print("Dataset shape:", df.shape)
print(df.head())

# ------------------------------
# 2. FIX COLUMN NAME ISSUE
# ------------------------------
# CSV column is 's', rename it to 'date'
df.rename(columns={'s': 'date'}, inplace=True)

# ------------------------------
# 3. Data Cleaning
# ------------------------------
df['date'] = pd.to_datetime(df['date'])
df['datetime'] = pd.to_datetime(df['datetime'])

# Fill missing card values
df['card'] = df['card'].fillna("CASH_USER")

# ------------------------------
# 4. Feature Engineering
# ------------------------------
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.dayofweek   # 0 = Monday
df['hour'] = df['datetime'].dt.hour

# ------------------------------
# 5. Exploratory Data Analysis
# ------------------------------

# Revenue by coffee type
plt.figure(figsize=(8,4))
df.groupby('coffee_name')['money'].sum().sort_values().plot(kind='barh')
plt.title("Revenue by Coffee Type")
plt.xlabel("Revenue")
plt.show()

# Monthly sales trend
plt.figure(figsize=(8,4))
df.groupby(df['date'].dt.to_period('M')).size().plot(marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales Count")
plt.show()

# Hourly sales distribution
plt.figure(figsize=(8,4))
sns.countplot(x='hour', data=df)
plt.title("Hourly Sales Distribution")
plt.show()

# Payment method distribution
plt.figure(figsize=(5,4))
df['cash_type'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title("Payment Method Distribution")
plt.ylabel("")
plt.show()

# ------------------------------
# 6. Machine Learning
# ------------------------------
X = df[['month', 'day', 'hour']]
y = df['money']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ------------------------------
# 7. Model Evaluation
# ------------------------------
print("\nModel Performance:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# ------------------------------
# 8. Sample Prediction
# ------------------------------
sample_input = pd.DataFrame({
    'month': [8],
    'day': [2],
    'hour': [10]
})

print("\nPredicted Sale Amount:",
      round(model.predict(sample_input)[0], 2))

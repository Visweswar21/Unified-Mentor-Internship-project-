
# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# 2. Load Dataset
df = pd.read_csv("C:/Users/viswe/Downloads/daily_transactions_sample.csv")

print("First 5 rows:")
print(df.head())

print("\nShape of dataset:")
print(df.shape)

print("\nDataset Info:")
print(df.info())

# 3. Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# 4. Data Cleaning

# Fill missing values
df["Subcategory"] = df["Subcategory"].fillna("Unknown")
df["Note"] = df["Note"].fillna("No Note")

# Convert Date column (fix date format)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Convert amount
df["Amount"] = df["Amount"].astype(float)

print("\nCleaned Data Info:")
print(df.info())

# 5. Summary Statistics
print("\nSummary Statistics")
print(df.describe())

# 6. Payment Mode Analysis
print("\nPayment Mode Counts:")
print(df["Mode"].value_counts())

plt.figure(figsize=(8,5))
sns.countplot(data=df, x="Mode", order=df["Mode"].value_counts().index)
plt.title("Payment Modes")
plt.xticks(rotation=45)
plt.show()

# 7. Category Analysis
print("\nCategory Counts:")
print(df["Category"].value_counts())

plt.figure(figsize=(8,5))
sns.countplot(data=df, x="Category",
              order=df["Category"].value_counts().index)
plt.title("Transaction Categories")
plt.xticks(rotation=45)
plt.show()

# 8. Subcategory Analysis
plt.figure(figsize=(10,6))
sns.countplot(data=df,
              x="Subcategory",
              order=df["Subcategory"].value_counts().index[:10])
plt.title("Top Subcategories")
plt.xticks(rotation=90)
plt.show()

# 9. Income vs Expense
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Income/Expense")
plt.title("Income vs Expense")
plt.show()

# 10. Transaction Amount Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Amount"], bins=20, kde=True)
plt.title("Distribution of Transaction Amount")
plt.show()

# 11. Category vs Amount
plt.figure(figsize=(10,6))
sns.boxplot(data=df,
            x="Amount",
            y="Category")
plt.title("Amount Distribution by Category")
plt.show()

# 12. Time Series Analysis

# Monthly transactions
monthly_data = df.resample("M", on="Date").sum(numeric_only=True)

plt.figure(figsize=(10,5))
plt.plot(monthly_data.index, monthly_data["Amount"], marker="o")
plt.title("Monthly Transaction Amount")
plt.xlabel("Month")
plt.ylabel("Amount")
plt.grid(True)
plt.show()

# Daily transactions
daily_data = df.groupby(df["Date"].dt.date).sum(numeric_only=True)

plt.figure(figsize=(10,5))
plt.plot(daily_data.index, daily_data["Amount"])
plt.title("Daily Transaction Amount")
plt.xlabel("Date")
plt.ylabel("Amount")
plt.xticks(rotation=45)
plt.show()

# 13. Correlation Analysis
pivot_table = df.pivot_table(
    index="Date",
    columns="Category",
    values="Amount",
    aggfunc="sum",
    fill_value=0
)

corr_matrix = pivot_table.corr()

plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 14. Scatter Plot
plt.figure(figsize=(7,5))
sns.scatterplot(data=df,
                x="Income/Expense",
                y="Mode")
plt.title("Income/Expense vs Payment Mode")
plt.show()

# 15. Final Insights
print("\n===== FINAL INSIGHTS =====")

print("Total Transactions:", len(df))
print("Total Amount:", df["Amount"].sum())
print("Average Transaction:", df["Amount"].mean())

print("\nTop Category:")
print(df["Category"].value_counts().head(1))

print("\nTop Payment Mode:")
print(df["Mode"].value_counts().head(1))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("C:/Users/chara/Downloads/climate_nasa.csv")

# Clean column names (safety)
df.columns = df.columns.str.strip()

print("Columns in dataset:")
print(df.columns)

# -----------------------------
# 2. Handle Missing Values
# -----------------------------
df = df.dropna(subset=['text'])

# -----------------------------
# 3. Download NLTK Resources
# -----------------------------
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# 4. Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)

# -----------------------------
# 5. Sentiment Analysis Function
# -----------------------------
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df['Sentiment'] = df['text'].apply(get_sentiment)

# -----------------------------
# 6. Feature Extraction (TF-IDF)
# -----------------------------
X = df['clean_text']
y = df['Sentiment']

vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# -----------------------------
# 7. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# -----------------------------
# 8. Train Model
# -----------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# -----------------------------
# 9. Prediction & Evaluation
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# -----------------------------
# 10. Save Output CSV
# -----------------------------
df.to_csv("climate_output_with_sentiment.csv", index=False)
print("Output saved as climate_output_with_sentiment.csv")

# -----------------------------
# 11. Visualization
# -----------------------------
df['Sentiment'].value_counts().plot(kind='bar')
plt.title("Sentiment Analysis of Climate Change Comments")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

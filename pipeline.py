import pandas as pd
from textblob import TextBlob
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.metrics import classification_report

RAW_PATH = '/content/netflix_titles.csv'
PROCESSED_PATH = '/content/netflix_titles.csv'
LOG_PATH = 'logs/mood_log.txt'

def log(message):
    os.makedirs("logs", exist_ok=True)
    with open(LOG_PATH, 'a', encoding="utf-8") as f:
        f.write(f"{datetime.now()} - {message}\n")
    print(message)

def extract_data(path):
    log("📥 Extracting Netflix data...")
    return pd.read_csv(path)

def label_mood(description):
    try:
        sentiment = TextBlob(description).sentiment.polarity
        if sentiment > 0.3:
            return 'Happy'
        elif sentiment < -0.1:
            return 'Dark'
        else:
            return 'Neutral'
    except:
        return 'Unknown'

def transform_data(df):
    log("🔧 Cleaning data and labeling mood...")
    df = df[['title', 'type', 'description', 'listed_in']].dropna(subset=['description'])
    df.rename(columns={'listed_in': 'genre'}, inplace=True)
    df['mood'] = df['description'].apply(label_mood)
    df = df[df['mood'] != 'Unknown']
    return df

def train_classifier(df):
    log("🧠 Training mood classifier...")
    X_train, X_test, y_train, y_test = train_test_split(df['description'], df['mood'], test_size=0.2, random_state=42)

    model = SKPipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', MultinomialNB())
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    log("📊 Classification Report:\n" + classification_report(y_test, y_pred))
    return model

def predict_mood(df, model):
    log("🔮 Predicting moods using trained model...")
    df['predicted_mood'] = model.predict(df['description'])
    return df

def load_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log("💾 Saving processed data...")
    df.to_csv(path, index=False)
    log(f"✅ Data saved to {path}")

def run_etl():
    try:
        df = extract_data(RAW_PATH)
        df_clean = transform_data(df)
        model = train_classifier(df_clean)
        df_final = predict_mood(df_clean, model)
        load_data(df_final, PROCESSED_PATH)
        log("🎉 MoodMatch ML ETL completed!\n")
    except Exception as e:
        log(f"❌ Error: {str(e)}")

if __name__ == '__main__':
    run_etl()

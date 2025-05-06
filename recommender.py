import pandas as pd
import random
import os

DATA_PATH = '/content/netflix_titles.csv'

def load_data():
    if not os.path.exists(DATA_PATH):
        print("❗ Please run the ETL pipeline first (pipeline_ml.py).")
        return None
    return pd.read_csv(DATA_PATH)

def recommend_shows(df, mood):
    mood = mood.lower()
    if 'predicted_mood' in df.columns:
        mood_df = df[df['predicted_mood'].str.lower() == mood]
    else:
        mood_df = df[df['mood'].str.lower() == mood]
    
    if mood_df.empty:
        print(f"😔 Sorry, no shows found for mood: {mood.capitalize()}")
        return
    
    print(f"\n🎬 Recommended Shows for your '{mood.capitalize()}' mood:\n")
    top_shows = mood_df.sample(n=min(5, len(mood_df)))
    
    for _, row in top_shows.iterrows():
        print(f"📌 {row['title']} ({row['type']})")
        print(f"📝 {row['description'][:150]}...")
        print()

def main():
    print("💡 Welcome to MoodMatch Recommender!")
    print("Available moods: Happy, Neutral, Dark")
    user_mood = input("Enter your mood: ").strip()
    
    df = load_data()
    if df is not None:
        recommend_shows(df, user_mood)

if __name__ == '__main__':
    main()

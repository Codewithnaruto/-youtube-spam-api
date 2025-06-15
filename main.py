import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

df = pd.read_csv("Youtube-Spam-Dataset.csv")

X = df["CONTENT"].astype(str)
y = df["CLASS"]
X_train, _, y_train, _ = train_test_split(X, y, stratify=y, random_state=42)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)
joblib.dump(model, "model.pkl")

known_authors = df["AUTHOR"].dropna().unique().tolist()
joblib.dump(known_authors, "known_authors.pkl")

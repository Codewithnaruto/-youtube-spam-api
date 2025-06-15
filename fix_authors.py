import pandas as pd
import joblib

# Load your dataset
df = pd.read_csv("Youtube-Spam-Dataset.csv")

# Extract unique author names
known_authors = df["AUTHOR"].dropna().unique().tolist()

# Save in clean format
joblib.dump(known_authors, "known_authors.pkl")

print("✅ known_authors.pkl saved successfully!")

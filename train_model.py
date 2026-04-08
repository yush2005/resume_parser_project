import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from utils.text_cleaning import clean_text

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("data/UpdatedResumeDataSet.csv")

# Clean text
df["cleaned_resume"] = df["Resume"].apply(clean_text)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Category"])

X = df["cleaned_resume"]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Build strong pipeline
# -----------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=15000,
        ngram_range=(1,3),
        stop_words="english",
        min_df=2
    )),
    ("clf", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# -----------------------------
# Save model
# -----------------------------
os.makedirs("models", exist_ok=True)

pickle.dump(pipeline, open("models/pipeline.pkl", "wb"))
pickle.dump(label_encoder, open("models/label_encoder.pkl", "wb"))

print("Model saved successfully!")
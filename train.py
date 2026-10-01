import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

nltk.download("stopwords")
nltk.download("wordnet")

# ---------- Dataset load ----------
data = pd.read_csv(r"c:\Users\gabha\Downloads\UpdatedResumeDataSet.csv")

# ---------- Cleaning ----------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

data["Cleaned_Resume"] = data["Resume_Text"].apply(clean_text)

# ---------- Category banana ----------
def get_category(skills):
    skills = skills.lower()
    if "machine learning" in skills or "data science" in skills or "nlp" in skills:
        return "Data Science"
    elif "java" in skills or "html" in skills or "css" in skills:
        return "Web/Java Developer"
    else:
        return "Other"

data["Category"] = data["Skills"].apply(get_category)

# ---------- TF-IDF ----------
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(data["Cleaned_Resume"])
y = data["Category"]

# ---------- Model train ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))

# ---------- Save karo (website isi ko use karegi) ----------
joblib.dump(model, "resume_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
data.to_csv("cleaned_resumes.csv", index=False)
print("Model, TF-IDF aur cleaned data save ho gaye")
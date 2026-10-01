import streamlit as st
import pandas as pd
import joblib
import re
import nltk
import pdfplumber
import docx
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords")
nltk.download("wordnet")

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

@st.cache_resource
def load_all():
    model = joblib.load("resume_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    data = pd.read_csv("cleaned_resumes.csv")
    data["Cleaned_Resume"] = data["Cleaned_Resume"].fillna("")
    return model, tfidf, data

model, tfidf, data = load_all()

def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + " "
    return text

def read_docx(file):
    d = docx.Document(file)
    return " ".join(p.text for p in d.paragraphs)

def rank(names, cleaned_texts, job_desc):
    job_vec = tfidf.transform([clean_text(job_desc)])
    res_vec = tfidf.transform(cleaned_texts)
    sim = cosine_similarity(job_vec, res_vec).flatten()
    job_cat = model.predict(job_vec)[0]
    idx = list(model.classes_).index(job_cat)
    prob = model.predict_proba(res_vec)[:, idx]
    final = 0.6 * sim + 0.4 * prob
    out = pd.DataFrame({
        "Candidate": names,
        "Similarity %": (sim * 100).round(1),
        "ML Match %": (prob * 100).round(1),
        "Final Score %": (final * 100).round(1),
    })
    out = out.sort_values("Final Score %", ascending=False).reset_index(drop=True)
    out.index = out.index + 1
    return job_cat, out

st.title("AI-Based Resume Screening System")
st.write("Job description likho, resumes do, aur best candidates ki ranking dekho.")

job_desc = st.text_area("Job Description", height=150,
    placeholder="Example: Python developer with machine learning, NLP and SQL skills")

mode = st.radio("Resumes kahan se lein?", ["Dataset ke resumes", "PDF/DOCX upload karo"])

uploaded = None
if mode == "PDF/DOCX upload karo":
    uploaded = st.file_uploader("Resumes upload karo", type=["pdf", "docx"], accept_multiple_files=True)

if st.button("Rank Candidates"):
    if not job_desc.strip():
        st.warning("Pehle job description likho.")
    elif mode == "Dataset ke resumes":
        job_cat, result = rank(data["Name"].tolist(), data["Cleaned_Resume"].tolist(), job_desc)
        st.success("Predicted Job Category: " + job_cat)
        st.dataframe(result)
        st.bar_chart(result.set_index("Candidate")["Final Score %"])
    else:
        if not uploaded:
            st.warning("Kam se kam ek resume upload karo.")
        else:
            names, texts = [], []
            for f in uploaded:
                raw = read_pdf(f) if f.name.lower().endswith(".pdf") else read_docx(f)
                names.append(f.name)
                texts.append(clean_text(raw))
            job_cat, result = rank(names, texts, job_desc)
            st.success("Predicted Job Category: " + job_cat)
            st.dataframe(result)
            st.bar_chart(result.set_index("Candidate")["Final Score %"])
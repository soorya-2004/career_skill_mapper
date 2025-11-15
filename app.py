import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define a list of known skills (customize as needed)
KNOWN_SKILLS = [
    "python", "java", "c++", "c", "javascript", "react", "angular",
    "django", "flask", "sql", "mongodb", "aws", "azure", "git",
    "linux", "tensorflow", "keras", "pytorch", "html", "css",
    "docker", "kubernetes", "spark", "hadoop", "pandas", "numpy",
    "matplotlib", "scikit-learn", "nlp", "excel", "powerbi", "tableau"
]
from bs4 import BeautifulSoup

def extract_skills_from_html(description_html):
    soup = BeautifulSoup(description_html, 'html.parser')
    text = soup.get_text(separator=' ').lower()
    found_skills = [skill for skill in KNOWN_SKILLS if skill in text]
    return list(set(found_skills))

# Load dataset
df = pd.read_csv("data/indeed.csv")

# Preprocess data
df["Job Description"].fillna("", inplace=True)
df["Job Title"].fillna("Unknown Role", inplace=True)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english")
job_desc_tfidf = vectorizer.fit_transform(df["Job Description"])

# Streamlit UI
st.title("🎯 Career Skill Mapper")
st.write("Get job role suggestions based on your skills or explore the skills required for a specific job role.")

option = st.radio("Choose an option", [
    "Get job roles based on your skills",
    "Get skills required for a job role"
])

# ✅ OPTION 1: Get job roles based on user's skills
if option == "Get job roles based on your skills":
    skills = st.text_input("Enter your skills (comma-separated):")

    if st.button("Suggest Roles"):
        if skills:
            skill_vec = vectorizer.transform([skills])
            similarities = cosine_similarity(skill_vec, job_desc_tfidf).flatten()
            top_indices = similarities.argsort()[-5:][::-1]
            matched_jobs = df.iloc[top_indices]

            st.subheader("🔍 Suggested Job Roles:")
            for index, row in matched_jobs.iterrows():
                job_title = row["Job Title"]
                description = row["Job Description"]
                st.subheader(f"🔹 {job_title}")
                skills_found = extract_skills_from_html(description)
                if skills_found:
                    st.write("🧠 Required Skills:", ", ".join(skills_found))
                else:
                    st.write("🧠 Required Skills: Not clearly mentioned.")
        else:
            st.warning("Please enter at least one skill.")

# ✅ OPTION 2: Get required skills based on a job role
elif option == "Get skills required for a job role":
    job = st.text_input("Enter a job role:")

    if st.button("Suggest Skills"):
        if job:
            job_vec = vectorizer.transform([job])
            similarities = cosine_similarity(job_vec, job_desc_tfidf).flatten()
            top_index = similarities.argmax()
            job_title = df.iloc[top_index]["Job Title"]
            job_description = df.iloc[top_index]["Job Description"]

            st.subheader(f"🔍 Closest Match: {job_title}")
            skills_found = extract_skills_from_html(job_description)
            if skills_found:
                st.write("🧠 Required Skills:", ", ".join(skills_found))
            else:
                st.write("🧠 Required Skills: Not clearly mentioned.")
        else:
            st.warning("Please enter a job role.")

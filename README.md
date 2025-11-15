# career_skill_mapper

This repository contains a small Streamlit app that maps user skills to job roles using TF-IDF and cosine similarity over a job listings dataset.

Files of interest
- `app.py` — Streamlit app entrypoint.
- `data/indeed.csv` — job listings dataset (not present in the repo because of the size constraints).
- `resume.txt` — example resume text (already present in the repo).

Quick start (Windows PowerShell)

1. Activate the virtual environment

```powershell
& .\jdvenv\Scripts\Activate.ps1
```

2. Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

3. Run the app (Streamlit)

```powershell
streamlit run .\app.py
```

4. Open your browser at http://localhost:8501

Notes
- If you cannot access the internet from this machine, install packages on another machine and copy wheels as described in the project docs or use `pip download`.
- If you see errors about `nltk` corpora, run `python -m nltk.downloader punkt wordnet stopwords`.

Repository on GitHub
Push your local repo to: `https://github.com/soorya-2004/career_skill_mapper`

If you want, I can prepare a commit message and exact git commands to run locally to push everything to your GitHub repo.

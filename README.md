
#  AI-Powered Resume Screener

An **AI-powered tool** that helps recruiters automatically screen resumes against a given job description.
This project reduces HR workload by analyzing resumes (manual uploads or Gmail attachments) and scoring them based on **semantic similarity**.

---

##  Features

* **Manual Upload**: Upload one or multiple PDF resumes and analyze them.
* **Gmail Fetch**: Securely fetch PDF resumes from Gmail (OAuth 2.0 authentication).
* **AI Scoring**: Uses `SentenceTransformer` embeddings (`all-MiniLM-L6-v2`) to compute semantic similarity.
* **Visualization**: Interactive **Plotly charts** to display candidate scores.
* **Shortlisting**: Adjustable threshold to automatically shortlist top candidates.
* **Streamlit UI**: Simple and modern web interface.

---

##  Demo Flow

1. Paste the **job description**.
2. Upload resumes **or** click **Fetch from Gmail**.
3. The app:

   * Extracts text from resumes (via `pdfplumber`).
   * Scores each resume vs. job description.
   * Displays results in a **table + bar chart**.
4. Candidates above your **threshold** are shortlisted automatically.

---

## 📂 Project Structure

```
├── resume_screener.py        # Main Streamlit app
├── requirements.txt          # Dependencies
├── client_secret.json        # Google API OAuth client (keep private!)
├── token.json                # Saved token after first login
├── README.md                 # Documentation
```

---

##  Installation

1. Clone this repo:

   ```bash
   git clone https://github.com/yourusername/ai-resume-screener.git
   cd ai-resume-screener
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Add your **Google API OAuth credentials** as `client_secret.json` (download from Google Cloud Console).

---

##  Running the App

Run with Streamlit:

```bash
streamlit run resume_screener.py
```

---

##  Gmail Integration Setup

1. Go to **Google Cloud Console** → Create OAuth 2.0 Client ID.
2. Set your **Streamlit app URL** as the Redirect URI ( no trailing `/`).
3. Download credentials JSON → save as `client_secret.json`.
4. On first run, authorize Gmail → a `token.json` will be created.

---

##  Dependencies

```txt
streamlit
pdfplumber
pandas
plotly
sentence-transformers
torch
google-auth
google-auth-oauthlib
google-auth-httplib2
google-api-python-client
```

---

##  Future Enhancements

* OCR for scanned resumes (Tesseract integration).
* Automated Gmail fetching on schedule.
* Support for Word docs (`.docx`).
* Enhanced keyword analysis (skills, education, experience extraction).

---

## 👨‍💻 Author

Developed by **Mudasir Malik** for Machine Learning Exhibition Project 

import os
import io
import base64
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PyPDF2 import PdfReader
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# -------------------- CONFIG --------------------
st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# -------------------- LOAD MODEL --------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------- FUNCTIONS --------------------
def gmail_authenticate():
    """Authenticate with Gmail API using credentials from Streamlit secrets."""
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Use OAuth flow
            flow = InstalledAppFlow.from_client_secrets_file(
                "client_secret.json", SCOPES
            )
            creds = flow.run_console()  # Use console for Streamlit Cloud
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)

def fetch_resumes_from_gmail():
    """Fetch PDF resumes from Gmail inbox."""
    service = gmail_authenticate()
    results = service.users().messages().list(userId="me", q="has:attachment filename:pdf").execute()
    messages = results.get("messages", [])
    resumes = []

    for msg in messages[:5]:  # Limit to 5 for speed
        msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
        for part in msg_data["payload"].get("parts", []):
            if part.get("filename", "").lower().endswith(".pdf"):
                att_id = part["body"]["attachmentId"]
                att = service.users().messages().attachments().get(userId="me", messageId=msg["id"], id=att_id).execute()
                file_data = base64.urlsafe_b64decode(att["data"])
                resumes.append(io.BytesIO(file_data))
    return resumes

def extract_text_from_pdf(file):
    """Extract text from PDF."""
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

def score_resume(resume_text, job_desc):
    """Calculate similarity score."""
    embeddings1 = model.encode([resume_text], convert_to_tensor=True)
    embeddings2 = model.encode([job_desc], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings1, embeddings2).item()
    fuzz_score = fuzz.partial_ratio(resume_text.lower(), job_desc.lower()) / 100
    final_score = (similarity + fuzz_score) / 2
    return round(final_score * 100, 2)

def visualize_scores(scores_df):
    """Plot bar chart of scores."""
    fig = go.Figure(
        go.Bar(
            x=scores_df["Candidate"],
            y=scores_df["Score"],
            marker_color="skyblue",
            text=scores_df["Score"],
            textposition="auto"
        )
    )
    fig.update_layout(title="Resume Match Scores", yaxis_title="Match %", height=400)
    st.plotly_chart(fig, use_container_width=True)

# -------------------- UI --------------------
st.title("📄 AI-Powered Resume Screener")
job_desc = st.text_area("Paste the Job Description", height=150)

fetch_from_gmail = st.checkbox("📥 Fetch resumes from Gmail (PDF only)")

resumes = []
if fetch_from_gmail:
    st.info("Fetching resumes from Gmail...")
    try:
        resumes = fetch_resumes_from_gmail()
        st.success(f"Fetched {len(resumes)} resumes from Gmail.")
    except Exception as e:
        st.error(f"Error fetching from Gmail: {e}")

uploaded_files = st.file_uploader("Or Upload Resume PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    resumes.extend(uploaded_files)

if st.button("Analyze Resumes") and job_desc and resumes:
    scores = []
    for idx, file in enumerate(resumes, start=1):
        text = extract_text_from_pdf(file)
        score = score_resume(text, job_desc)
        scores.append({"Candidate": f"Resume {idx}", "Score": score})
    
    scores_df = pd.DataFrame(scores).sort_values(by="Score", ascending=False).reset_index(drop=True)
    visualize_scores(scores_df)
    
    best_candidate = scores_df.iloc[0]
    st.success(f"🏆 Best Candidate: {best_candidate['Candidate']} with {best_candidate['Score']}% match")
    
    st.dataframe(scores_df)
else:
    st.warning("Please provide a job description and at least one resume.")


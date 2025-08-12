import os
import io
import base64
import pdfplumber
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# --- Gmail API Settings ---
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# Load NLP model
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Gmail Authentication ---
def gmail_authenticate():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "client_secret.json", SCOPES
            )
            # Prevents browser errors in Streamlit Cloud
            creds = flow.run_local_server(port=0, open_browser=False)
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)

# --- Fetch PDF Resumes from Gmail ---
def fetch_resumes_from_gmail():
    service = gmail_authenticate()
    results = service.users().messages().list(userId="me", q="has:attachment filename:pdf").execute()
    messages = results.get("messages", [])

    resumes = []
    for msg in messages:
        msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
        for part in msg_data["payload"].get("parts", []):
            if part.get("filename", "").lower().endswith(".pdf"):
                att_id = part["body"]["attachmentId"]
                att = service.users().messages().attachments().get(
                    userId="me", messageId=msg["id"], id=att_id
                ).execute()
                data = base64.urlsafe_b64decode(att["data"])
                resumes.append(io.BytesIO(data))
    return resumes

# --- Extract Text from PDF ---
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# --- Calculate Match Score ---
def calculate_match(resume_text, job_description):
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    jd_emb = model.encode(job_description, convert_to_tensor=True)
    score = util.pytorch_cos_sim(resume_emb, jd_emb).item()
    return round(score * 100, 2)

# --- Visualize Match ---
def plot_match_chart(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Match Score"},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "green"}}
    ))
    return fig

# --- Streamlit UI ---
st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")
st.title("📄 AI-Powered Resume Screener")
st.write("Upload resumes or fetch from Gmail, and compare them with a job description.")

job_description = st.text_area("📌 Job Description", placeholder="Paste the job description here...")

col1, col2 = st.columns(2)

with col1:
    uploaded_files = st.file_uploader("📂 Upload Resumes (PDF only)", type=["pdf"], accept_multiple_files=True)

with col2:
    if st.button("📩 Fetch from Gmail"):
        try:
            uploaded_files = fetch_resumes_from_gmail()
            st.success(f"Fetched {len(uploaded_files)} resumes from Gmail.")
        except Exception as e:
            st.error(f"Error fetching from Gmail: {e}")

if uploaded_files and job_description:
    results = []
    for file in uploaded_files:
        resume_text = extract_text_from_pdf(file)
        score = calculate_match(resume_text, job_description)
        results.append({"filename": getattr(file, "name", "Gmail PDF"), "score": score})

    # Sort results
    results_df = pd.DataFrame(results).sort_values(by="score", ascending=False)

    # Show table
    st.dataframe(results_df)

    # Show best match
    best_match = results_df.iloc[0]
    st.subheader(f"🏆 Best Candidate: {best_match['filename']} ({best_match['score']}%)")

    # Plot chart
    st.plotly_chart(plot_match_chart(best_match["score"]), use_container_width=True)

else:
    st.info("Please provide a job description and upload or fetch resumes.")

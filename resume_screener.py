import os
import json
import base64
import io
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pdfplumber
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# -------------------------------
# CONFIG
# -------------------------------
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
CLIENT_SECRET_FILE = "client_secret.json"
TOKEN_FILE = "token.json"

# Load Sentence Transformer
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# -------------------------------
# GMAIL AUTH HELPERS
# -------------------------------
def get_google_auth_url():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRET_FILE,
        scopes=SCOPES,
        redirect_uri=st.secrets["redirect_uri"]
    )
    auth_url, _ = flow.authorization_url(prompt="consent")
    return auth_url

def exchange_code_for_tokens(code):
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRET_FILE,
        scopes=SCOPES,
        redirect_uri=st.secrets["redirect_uri"]
    )
    flow.fetch_token(code=code)
    creds = flow.credentials
    with open(TOKEN_FILE, "w") as token:
        token.write(creds.to_json())
    return creds

def load_credentials():
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        return creds
    return None

def fetch_resumes_from_gmail(creds):
    service = build("gmail", "v1", credentials=creds)
    results = service.users().messages().list(userId="me", q="has:attachment filename:pdf").execute()
    messages = results.get("messages", [])
    resumes = []

    for msg in messages:
        msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
        for part in msg_data.get("payload", {}).get("parts", []):
            if part["filename"].endswith(".pdf"):
                att_id = part["body"]["attachmentId"]
                att = service.users().messages().attachments().get(userId="me", messageId=msg["id"], id=att_id).execute()
                file_data = base64.urlsafe_b64decode(att["data"])
                resumes.append({"name": part["filename"], "data": file_data})
    return resumes

# -------------------------------
# RESUME PROCESSING
# -------------------------------
def extract_text_from_pdf(file_bytes):
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def calculate_match_score(resume_text, job_desc):
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    jd_embedding = model.encode(job_desc, convert_to_tensor=True)
    similarity = util.cos_sim(resume_embedding, jd_embedding)
    return round(float(similarity) * 100, 2)

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")
st.title("📄 AI Powered Resume Screener with Gmail Integration")

job_description = st.text_area("📌 Job Description", height=150)

# HR threshold input
threshold = st.slider("📊 Minimum Score to Shortlist Candidate", 0, 100, 75)

code_param = st.query_params.get("code", [None])[0]
creds = load_credentials()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Manual Resume Upload")
    uploaded_files = st.file_uploader("Upload Resumes (PDF only)", type="pdf", accept_multiple_files=True)

with col2:
    st.subheader("📧 Fetch Resumes from Gmail")
    if creds:
        st.success("✅ Authorized with Google")
        if st.button("Fetch from Gmail"):
            gmail_resumes = fetch_resumes_from_gmail(creds)
            for gr in gmail_resumes:
                uploaded_files.append(io.BytesIO(gr["data"]))
            st.success(f"Fetched {len(gmail_resumes)} resumes from Gmail!")
    else:
        if code_param:
            creds = exchange_code_for_tokens(code_param)
            st.query_params.clear()
            st.success("✅ Authorized successfully! You can now fetch resumes.")
        else:
            auth_url = get_google_auth_url()
            st.markdown(f"[🔑 Authorize with Google]({auth_url})")

if st.button("Run Screening"):
    if not job_description:
        st.error("Please enter a job description before screening.")
    elif not uploaded_files:
        st.error("Please upload at least one resume.")
    else:
        results = []
        for file in uploaded_files:
            if isinstance(file, io.BytesIO):
                resume_bytes = file.read()
                name = "Gmail Resume"
            else:
                resume_bytes = file.read()
                name = file.name

            resume_text = extract_text_from_pdf(resume_bytes)
            score = calculate_match_score(resume_text, job_description)
            results.append({"Name": name, "Score": score})

        # Create DataFrame
        df = pd.DataFrame(results).sort_values(by="Score", ascending=False)
        st.subheader("📊 Match Results")
        st.dataframe(df)

        # Plot all scores
        fig = go.Figure([go.Bar(x=df["Name"], y=df["Score"], text=df["Score"], textposition='auto')])
        fig.update_layout(title="Resume Match Scores", yaxis=dict(title="Score (%)"))
        st.plotly_chart(fig, use_container_width=True)

        # Shortlisted Candidates
        shortlisted = df[df["Score"] >= threshold]
        st.subheader(f"✅ Shortlisted Candidates (Score ≥ {threshold}%)")
        if not shortlisted.empty:
            st.dataframe(shortlisted)
            st.success(f"🏆 Best Match: **{shortlisted.iloc[0]['Name']}** with a score of {shortlisted.iloc[0]['Score']}%")
        else:
            st.warning("No candidates met the threshold criteria.")

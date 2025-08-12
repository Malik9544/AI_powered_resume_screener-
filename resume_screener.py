import os
import json
import base64
import pdfplumber
import pandas as pd
import streamlit as st
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# ------------------ CONFIG ------------------
st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------ GMAIL AUTH ------------------
def gmail_authenticate():
    creds_json = st.secrets["GMAIL_CREDENTIALS"]  # Load from Streamlit Secrets
    with open("gmail_credentials.json", "w") as f:
        f.write(creds_json)

    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("gmail_credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)

# ------------------ FETCH RESUMES FROM GMAIL ------------------
def fetch_resumes_from_gmail():
    service = gmail_authenticate()
    resumes = []

    try:
        results = service.users().messages().list(userId='me', q="has:attachment filename:pdf").execute()
        messages = results.get('messages', [])
        for msg in messages:
            msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
            for part in msg_data['payload'].get('parts', []):
                if part['filename'].endswith('.pdf'):
                    att_id = part['body']['attachmentId']
                    att = service.users().messages().attachments().get(userId='me', messageId=msg['id'], id=att_id).execute()
                    file_data = base64.urlsafe_b64decode(att['data'])
                    resumes.append({"filename": part['filename'], "file": BytesIO(file_data)})
    except HttpError as error:
        st.error(f"An error occurred: {error}")

    return resumes

# ------------------ RESUME PARSING ------------------
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return " ".join(page.extract_text() or "" for page in pdf.pages)

# ------------------ MATCHING ------------------
def calculate_similarity(resume_text, jd_text):
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    return float(util.pytorch_cos_sim(resume_emb, jd_emb)[0][0])

# ------------------ UI ------------------
st.title("📄 AI-Powered Resume Screener")

jd_text = st.text_area("📝 Job Description", placeholder="Paste the job description here...")

uploaded_files = st.file_uploader("📂 Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

if st.button("📥 Fetch from Gmail"):
    gmail_resumes = fetch_resumes_from_gmail()
    for r in gmail_resumes:
        uploaded_files.append(r["file"])

if jd_text and uploaded_files:
    results = []
    for file in uploaded_files:
        resume_text = extract_text_from_pdf(file)
        score = calculate_similarity(resume_text, jd_text) * 100
        results.append({"filename": file.name, "score": score})

    df = pd.DataFrame(results).sort_values(by="score", ascending=False)
    st.subheader("📊 Resume Match Scores")
    st.dataframe(df)

    best_match = df.iloc[0]
    st.success(f"🏆 Best Match: {best_match['filename']} ({best_match['score']:.2f}%)")

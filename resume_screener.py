import os
import io
import pdfplumber
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer, util
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import base64

# ===============================
# CONFIG
# ===============================
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit UI settings
st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")
st.title("📄 AI-Powered Resume Screener")
st.write("Automated Resume Screening with NLP + Gmail API")

# ===============================
# GMAIL API HELPER FUNCTIONS
# ===============================
def gmail_authenticate():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "client_secret_1312624875-ppqs8a1ufg2ed6s7f6p0ltn85pbd3kbh.apps.googleusercontent.com.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

def fetch_pdfs_from_gmail(service, max_results=5):
    results = service.users().messages().list(userId='me', maxResults=max_results).execute()
    messages = results.get('messages', [])
    pdf_files = []
    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
        for part in msg_data.get('payload', {}).get('parts', []):
            if part.get('filename', '').lower().endswith('.pdf'):
                att_id = part['body']['attachmentId']
                att = service.users().messages().attachments().get(userId='me', messageId=msg['id'], id=att_id).execute()
                file_data = base64.urlsafe_b64decode(att['data'])
                file_path = os.path.join("temp_resumes", part['filename'])
                os.makedirs("temp_resumes", exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(file_data)
                pdf_files.append(file_path)
    return pdf_files

# ===============================
# RESUME PROCESSING
# ===============================
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def score_resume(resume_text, job_description):
    resume_emb = MODEL.encode(resume_text, convert_to_tensor=True)
    job_emb = MODEL.encode(job_description, convert_to_tensor=True)
    similarity = util.cos_sim(resume_emb, job_emb).item()
    return round(similarity * 100, 2)

def visualize_scores(scores_dict):
    names = list(scores_dict.keys())
    scores = list(scores_dict.values())
    fig = go.Figure(data=[go.Bar(x=names, y=scores, text=scores, textposition='auto')])
    fig.update_layout(title="Resume Match Scores", xaxis_title="Candidate", yaxis_title="Match %")
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# UI
# ===============================
job_description = st.text_area("📌 Enter Job Description", height=150)

option = st.radio("Select Resume Source:", ("📂 Manual Upload", "📧 Fetch from Gmail"))

resumes_texts = {}
if option == "📂 Manual Upload":
    uploaded_files = st.file_uploader("Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)
    if uploaded_files and st.button("Process Resumes"):
        for file in uploaded_files:
            resume_text = extract_text_from_pdf(file)
            resumes_texts[file.name] = resume_text

elif option == "📧 Fetch from Gmail":
    num_files = st.slider("Number of resumes to fetch", 1, 20, 5)
    if st.button("Fetch from Gmail"):
        service = gmail_authenticate()
        pdf_paths = fetch_pdfs_from_gmail(service, num_files)
        for path in pdf_paths:
            resumes_texts[os.path.basename(path)] = extract_text_from_pdf(path)
        st.success(f"Fetched {len(pdf_paths)} PDF resumes from Gmail!")

# ===============================
# SCORING & RESULTS
# ===============================
if resumes_texts and job_description:
    scores = {name: score_resume(text, job_description) for name, text in resumes_texts.items()}
    st.subheader("📊 Match Results")
    visualize_scores(scores)
    best_candidate = max(scores, key=scores.get)
    st.success(f"🏆 Best Match: **{best_candidate}** ({scores[best_candidate]}%)")

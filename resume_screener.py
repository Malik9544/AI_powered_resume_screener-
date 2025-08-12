import streamlit as st
import pdfplumber
import pandas as pd
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import base64
import os

# ------------------------
# CONFIG
# ------------------------
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
MODEL = SentenceTransformer('all-MiniLM-L6-v2')

st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")

# ------------------------
# FUNCTIONS
# ------------------------
def gmail_authenticate():
    """Authenticate Gmail API in a cloud-friendly way."""
    token_file = "token.json"
    creds = None

    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "client_secret.json", SCOPES
            )
            # Cloud-friendly: run_console instead of run_local_server
            auth_url, _ = flow.authorization_url(prompt='consent')
            st.markdown(f"### [Click here to authorize Gmail access]({auth_url})")
            auth_code = st.text_input("Enter the authorization code from Google here:")

            if auth_code:
                flow.fetch_token(code=auth_code)
                creds = flow.credentials
                with open(token_file, 'w') as token:
                    token.write(creds.to_json())
                st.success("✅ Gmail authorization successful!")
    return creds


def fetch_resumes_from_gmail():
    """Fetch PDF resumes from Gmail inbox."""
    creds = gmail_authenticate()
    if not creds:
        return []

    try:
        service = build('gmail', 'v1', credentials=creds)
        results = service.users().messages().list(userId='me', maxResults=10, q="has:attachment filename:pdf").execute()
        messages = results.get('messages', [])
        resumes = []

        for msg in messages:
            msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
            for part in msg_data['payload'].get('parts', []):
                if part['filename'].endswith('.pdf'):
                    att_id = part['body']['attachmentId']
                    att = service.users().messages().attachments().get(
                        userId='me', messageId=msg['id'], id=att_id
                    ).execute()
                    data = base64.urlsafe_b64decode(att['data'])
                    file_path = f"gmail_resume_{part['filename']}"
                    with open(file_path, "wb") as f:
                        f.write(data)
                    resumes.append(file_path)
        return resumes

    except Exception as e:
        st.error(f"❌ Error fetching from Gmail: {e}")
        return []


def extract_text_from_pdf(pdf_file):
    """Extracts all text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


def score_resume(resume_text, job_desc):
    """Scores resume similarity to job description."""
    embeddings = MODEL.encode([resume_text, job_desc], convert_to_tensor=True)
    score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return round(score * 100, 2)


def visualize_scores(results):
    """Creates a Plotly bar chart of resume scores."""
    fig = go.Figure(data=[
        go.Bar(x=[r['name'] for r in results], y=[r['score'] for r in results], marker_color='royalblue')
    ])
    fig.update_layout(title="Resume Match Scores", xaxis_title="Resume", yaxis_title="Score (%)", height=400)
    st.plotly_chart(fig, use_container_width=True)


# ------------------------
# STREAMLIT UI
# ------------------------
st.title("📄 AI-Powered Resume Screener")
st.write("Upload resumes or fetch from Gmail. Set your match criteria and find the best candidates automatically.")

job_description = st.text_area("📌 Enter Job Description", height=150)
threshold = st.slider("Set minimum match % to shortlist", 0, 100, 70)

col1, col2 = st.columns(2)

with col1:
    uploaded_files = st.file_uploader("📤 Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)
    analyze_button = st.button("Analyze Uploaded Resumes")

with col2:
    gmail_button = st.button("📧 Fetch from Gmail")

results = []

if analyze_button and uploaded_files:
    for uploaded_file in uploaded_files:
        resume_text = extract_text_from_pdf(uploaded_file)
        score = score_resume(resume_text, job_description)
        results.append({"name": uploaded_file.name, "score": score})

elif gmail_button:
    pdf_paths = fetch_resumes_from_gmail()
    for pdf in pdf_paths:
        resume_text = extract_text_from_pdf(pdf)
        score = score_resume(resume_text, job_description)
        results.append({"name": os.path.basename(pdf), "score": score})

if results:
    visualize_scores(results)

    st.subheader("📊 Match Results")
    for r in results:
        if r['score'] >= threshold:
            st.success(f"✅ {r['name']} - {r['score']}% (Shortlisted)")
        else:
            st.warning(f"⚠️ {r['name']} - {r['score']}%")

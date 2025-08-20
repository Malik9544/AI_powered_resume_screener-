# resume_screener.py
import os
import io
import base64
import streamlit as st
import pdfplumber
import pandas as pd
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer, util
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image

# ---------------- CONFIG ----------------
CLIENT_SECRET_FILE = "client_secret.json"
TOKEN_FILE = "token.json"
DEFAULT_REDIRECT_URI = "https://mxdkvunyvferw2lfjeihzr.streamlit.app"  # <- your app URL
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")
st.title("AI-Powered Resume Screener")

# ---------------- Model (cached) ----------------
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

MODEL = load_model()

# ---------------- Gmail auth helpers ----------------
def get_redirect_uri():
    if "redirect_uri" in st.secrets:
        return st.secrets["redirect_uri"]
    return DEFAULT_REDIRECT_URI

def load_saved_credentials():
    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
            return creds
        except Exception:
            return None
    return None

def save_credentials(creds):
    with open(TOKEN_FILE, "w") as tf:
        tf.write(creds.to_json())

def build_flow():
    if not os.path.exists(CLIENT_SECRET_FILE):
        st.error(f"Missing `{CLIENT_SECRET_FILE}` in app folder.")
        return None
    redirect_uri = get_redirect_uri()
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRET_FILE,
        scopes=SCOPES,
        redirect_uri=redirect_uri
    )
    return flow

def ensure_authorized():
    creds = load_saved_credentials()
    if creds and creds.valid:
        return creds
    flow = build_flow()
    if flow is None:
        return None
    params = st.query_params
    code = params.get("code")
    if code:
        try:
            flow.fetch_token(code=code)
            creds = flow.credentials
            save_credentials(creds)
            st.query_params.clear()
            st.success("✅ Google authorization complete — you can now fetch resumes.")
            return creds
        except Exception as e:
            st.error(f"Failed to exchange code for token: {e}")
            return None
    else:
        auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")
        st.markdown(f"[🔑 Click here to authorize Gmail access]({auth_url})")
        return None

def fetch_pdf_attachments_from_gmail(creds, max_messages=10, days_back=30, strict=False):
    service = build("gmail", "v1", credentials=creds)
    query = f"has:attachment filename:pdf newer_than:{days_back}d"
    if strict:
        query += " (resume OR cv)"
    try:
        resp = service.users().messages().list(
            userId="me", maxResults=max_messages, q=query
        ).execute()
        messages = resp.get("messages", []) or []
    except Exception as e:
        st.error(f"Gmail list error: {e}")
        return []
    attachments = []
    for msg in messages:
        try:
            msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
            parts = msg_data.get("payload", {}).get("parts", []) or []
            for part in parts:
                filename = part.get("filename", "")
                body = part.get("body", {})
                if filename and filename.lower().endswith(".pdf"):
                    att_id = body.get("attachmentId")
                    if not att_id:
                        continue
                    att = service.users().messages().attachments().get(
                        userId="me", messageId=msg["id"], id=att_id
                    ).execute()
                    data = base64.urlsafe_b64decode(att.get("data", ""))
                    attachments.append({"name": filename, "bytes": data})
        except Exception as e:
            st.warning(f"Failed to process one message: {e}")
            continue
    return attachments

# ---------------- Resume parsing & OCR ----------------
def extract_text_from_pdf_bytes(pdf_bytes):
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
            text = "\n".join(pages)
            if text.strip():
                return text
    except Exception:
        pass
    # Fallback to OCR
    try:
        images = convert_from_bytes(pdf_bytes)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img)
        return text
    except Exception:
        return ""

def calculate_semantic_score(resume_text, job_description):
    if not resume_text.strip():
        return 0.0
    emb_resume = MODEL.encode(resume_text, convert_to_tensor=True)
    emb_jd = MODEL.encode(job_description, convert_to_tensor=True)
    score = util.pytorch_cos_sim(emb_resume, emb_jd).item()
    return round(float(score) * 100, 2)

# ---------------- UI ----------------
st.markdown("**Paste the job description**, then either upload resumes or fetch from Gmail.")

job_description = st.text_area("Job description", height=180)
threshold = st.slider("Minimum match % to shortlist", 0, 100, 75)

# Manual upload
st.subheader("Manual resumes")
uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)
analyze_uploaded = st.button("Analyze uploaded resumes")

# Gmail fetch
st.subheader("Gmail fetch")
days_back = st.selectbox("Fetch resumes from last…", [7, 30, 90], index=1)
strict_filter = st.checkbox("Strict filtering (only resumes/CVs)", value=True)
num_to_fetch = st.number_input("Max messages to scan", min_value=1, max_value=50, value=10, step=1)
fetch_from_gmail = st.button("Fetch from Gmail")

results = []

# Manual analysis
if analyze_uploaded:
    if not job_description.strip():
        st.error("Paste job description first.")
    elif not uploaded_files:
        st.error("Upload at least one resume.")
    else:
        with st.spinner("Scoring resumes..."):
            for f in uploaded_files:
                pdf_bytes = f.read()
                resume_text = extract_text_from_pdf_bytes(pdf_bytes)
                score = calculate_semantic_score(resume_text, job_description)
                results.append({"name": f.name, "score": score})

# Gmail analysis
if fetch_from_gmail:
    if not job_description.strip():
        st.error("Paste job description first.")
    else:
        creds = load_saved_credentials()
        if creds and creds.valid:
            with st.spinner("Fetching PDFs from Gmail..."):
                attachments = fetch_pdf_attachments_from_gmail(
                    creds, max_messages=num_to_fetch, days_back=days_back, strict=strict_filter
                )
                if attachments:
                    with st.spinner("Scoring resumes..."):
                        for att in attachments:
                            txt = extract_text_from_pdf_bytes(att["bytes"])
                            sc = calculate_semantic_score(txt, job_description)
                            results.append({"name": att["name"], "score": sc})
                        st.success(f"Fetched and scored {len(attachments)} PDF(s).")
                else:
                    st.info("No matching resumes found.")
        else:
            creds2 = ensure_authorized()
            if creds2:
                with st.spinner("Fetching PDFs from Gmail..."):
                    attachments = fetch_pdf_attachments_from_gmail(
                        creds2, max_messages=num_to_fetch, days_back=days_back, strict=strict_filter
                    )
                    if attachments:
                        with st.spinner("Scoring resumes..."):
                            for att in attachments:
                                txt = extract_text_from_pdf_bytes(att["bytes"])
                                sc = calculate_semantic_score(txt, job_description)
                                results.append({"name": att["name"], "score": sc})
                        st.success(f"Fetched and scored {len(attachments)} PDF(s).")

# Results
if results:
    df = pd.DataFrame(results).sort_values(by="score", ascending=False).reset_index(drop=True)
    st.subheader("Match results")
    st.dataframe(df)

    # Plot results
    fig = go.Figure(go.Bar(x=df["name"], y=df["score"], text=df["score"], textposition="auto"))
    fig.update_layout(title="Resume Match Scores", xaxis_title="Candidate", yaxis_title="Match %", height=420)
    st.plotly_chart(fig, use_container_width=True)

    # Shortlist
    shortlisted = df[df["score"] >= threshold]
    st.subheader(f"Shortlisted (≥ {threshold}%) — {len(shortlisted)}")
    if not shortlisted.empty:
        st.table(shortlisted)
        # Download button
        csv = shortlisted.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Shortlisted CSV", csv, "shortlisted.csv", "text/csv")
        top = shortlisted.iloc[0]
        st.success(f"Top candidate: {top['name']} — {top['score']}%")
    else:
        st.warning("No candidates met the threshold.")

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

# ---------------- CONFIG ----------------
# Change these only if you renamed files or want a different redirect URL
CLIENT_SECRET_FILE = "client_secret.json"
TOKEN_FILE = "token.json"

# If you put redirect_uri in Streamlit Secrets (recommended), use it:
DEFAULT_REDIRECT_URI = "https://mxdkvunyvferw2lfjeihzr.streamlit.app"  # <- your app URL

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")
st.title("AI-Powered Resume Screener")

# ---------------- Model (cached) ----------------
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

MODEL = load_model()

# ---------------- Helpers: Gmail auth & fetch ----------------
def get_redirect_uri():
    # prefer user-supplied secret, else default
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
        st.error(f"Missing `{CLIENT_SECRET_FILE}` in app folder. Upload it (OAuth client JSON).")
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

    # Get query params from URL
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
        st.info("To fetch resumes from Gmail you must authorize this app to read your mailbox (read-only).")
        st.markdown(f"[🔑 Click here to authorize Gmail access]({auth_url})")
        st.caption("After authorizing, Google will redirect back here automatically.")
        return None

def fetch_pdf_attachments_from_gmail(creds, max_messages=10):
    """
    Returns list of dicts: {'name': filename, 'bytes': <bytes>}
    """
    service = build("gmail", "v1", credentials=creds)
    try:
        resp = service.users().messages().list(userId="me", maxResults=max_messages, q="has:attachment filename:pdf").execute()
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
                    att = service.users().messages().attachments().get(userId="me", messageId=msg["id"], id=att_id).execute()
                    data = base64.urlsafe_b64decode(att.get("data", ""))
                    attachments.append({"name": filename, "bytes": data})
        except Exception as e:
            # keep going if one message fails
            st.warning(f"Warning: failed to process one message: {e}")
            continue
    return attachments

# ---------------- Resume parsing & scoring ----------------
def extract_text_from_pdf_bytes(pdf_bytes):
    # pdfplumber works with BytesIO
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
            return "\n".join(pages)
    except Exception:
        # fallback: empty text (you can add OCR here later)
        return ""

def calculate_semantic_score(resume_text, job_description):
    if not resume_text.strip():
        return 0.0
    # encode both, compute cosine
    emb_resume = MODEL.encode(resume_text, convert_to_tensor=True)
    emb_jd = MODEL.encode(job_description, convert_to_tensor=True)
    score = util.pytorch_cos_sim(emb_resume, emb_jd).item()
    return round(float(score) * 100, 2)

# ---------------- UI ----------------
st.markdown("**Paste the job description**, then either upload resumes or fetch from Gmail. Both buttons are available below.")

job_description = st.text_area("Job description (paste full JD)", height=180)

threshold = st.slider("Minimum match % to shortlist", 0, 100, 75)

# File upload area (manual)
st.subheader("Manual resumes")
uploaded_files = st.file_uploader("Upload one or more PDF resumes", type=["pdf"], accept_multiple_files=True)
analyze_uploaded = st.button("Analyze uploaded resumes")

# Gmail area (always visible)
st.subheader("Gmail fetch")
st.markdown("Fetch recent PDF attachments from your Gmail (read-only).")
# Offer how many to fetch
num_to_fetch = st.number_input("Max number of messages to scan (recent)", min_value=1, max_value=50, value=10, step=1)
fetch_from_gmail = st.button("Fetch from Gmail")

results = []  # list of dicts {'name','score'}

# Manual analysis action
if analyze_uploaded:
    if not job_description.strip():
        st.error("Please paste a job description before analyzing.")
    elif not uploaded_files:
        st.error("Please upload at least one PDF resume.")
    else:
        with st.spinner("Scoring uploaded resumes..."):
            for f in uploaded_files:
                try:
                    pdf_bytes = f.read()
                    resume_text = extract_text_from_pdf_bytes(pdf_bytes)
                    score = calculate_semantic_score(resume_text, job_description)
                    results.append({"name": getattr(f, "name", "Uploaded PDF"), "score": score})
                except Exception as e:
                    st.warning(f"Failed to process {getattr(f,'name','uploaded')}: {e}")

# Gmail fetch action
if fetch_from_gmail:
    if not job_description.strip():
        st.error("Please paste a job description before fetching + analyzing.")
    else:
        creds = load_saved_credentials()
        if creds and creds.valid:
            # direct fetch
            with st.spinner("Fetching PDFs from Gmail..."):
                attachments = fetch_pdf_attachments_from_gmail(creds, max_messages=num_to_fetch)
                if not attachments:
                    st.info("No PDF attachments found in recent messages.")
                else:
                    with st.spinner("Scoring fetched resumes..."):
                        for att in attachments:
                            txt = extract_text_from_pdf_bytes(att["bytes"])
                            sc = calculate_semantic_score(txt, job_description)
                            results.append({"name": att["name"], "score": sc})
                        st.success(f"Fetched and scored {len(attachments)} PDF(s) from Gmail.")
        else:
            # not authorized yet (or token expired) -> present auth link + wait for redirect
            st.info("You must authorize the app to read your Gmail (read-only).")
            creds2 = ensure_authorized()
            if creds2:
                # After authorization completes, automatically fetch
                with st.spinner("Fetching PDFs from Gmail (post-auth)..."):
                    attachments = fetch_pdf_attachments_from_gmail(creds2, max_messages=num_to_fetch)
                    if not attachments:
                        st.info("No PDF attachments found in recent messages.")
                    else:
                        with st.spinner("Scoring fetched resumes..."):
                            for att in attachments:
                                txt = extract_text_from_pdf_bytes(att["bytes"])
                                sc = calculate_semantic_score(txt, job_description)
                                results.append({"name": att["name"], "score": sc})
                            st.success(f"Fetched and scored {len(attachments)} PDF(s) from Gmail.")


# Results display (if any)
if results:
    df = pd.DataFrame(results).sort_values(by="score", ascending=False).reset_index(drop=True)
    st.subheader("Match results")
    st.dataframe(df)

    # Plot top results
    fig = go.Figure(go.Bar(x=df["name"], y=df["score"], text=df["score"], textposition="auto"))
    fig.update_layout(title="Resume Match Scores", xaxis_title="Candidate", yaxis_title="Match %", height=420)
    st.plotly_chart(fig, use_container_width=True)

    # Shortlist section
    shortlisted = df[df["score"] >= threshold]
    st.subheader(f"Shortlisted (score ≥ {threshold}%) — {len(shortlisted)} candidate(s)")
    if not shortlisted.empty:
        st.table(shortlisted)
        top = shortlisted.iloc[0]
        st.success(f"Top shortlisted candidate: {top['name']} — {top['score']}%")
    else:
        st.warning("No candidates met the threshold.")

# small helpful note
st.info("Tip: If you authorize Gmail, Google will redirect back to this app URL with a code. Ensure your OAuth client in Google Cloud has the app URL set as a redirect URI (no trailing slash).")



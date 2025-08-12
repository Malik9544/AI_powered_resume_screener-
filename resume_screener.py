from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
import base64
import email
import pdfplumber
import os
import io
from PyPDF2 import PdfReader
import streamlit as st

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def gmail_authenticate():
    """Authenticate Gmail API on Streamlit Cloud using copy-paste code flow"""
    creds = None

    # Use Streamlit secrets for credentials
    if "gmail_creds" in st.secrets:
        with open("client_secret.json", "w") as f:
            f.write(st.secrets["gmail_creds"])

    # OAuth Flow without local browser
    flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
    auth_url, _ = flow.authorization_url(prompt='consent')

    st.markdown(f"**[Click here to authorize Gmail access]({auth_url})**")
    auth_code = st.text_input("Enter the authorization code from Google here:")

    if auth_code:
        flow.fetch_token(code=auth_code)
        creds = flow.credentials
        return build('gmail', 'v1', credentials=creds)
    else:
        return None


def fetch_resumes_from_gmail():
    """Fetch latest PDF resumes from Gmail"""
    service = gmail_authenticate()
    if not service:
        return []

    # Search for PDF attachments
    results = service.users().messages().list(
        userId='me',
        q="has:attachment filename:pdf"
    ).execute()

    messages = results.get('messages', [])
    pdf_texts = []

    for msg in messages[:5]:  # limit for testing
        msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
        for part in msg_data['payload'].get('parts', []):
            if part['filename'].endswith('.pdf'):
                attach_id = part['body']['attachmentId']
                attachment = service.users().messages().attachments().get(
                    userId='me', messageId=msg['id'], id=attach_id
                ).execute()
                file_data = base64.urlsafe_b64decode(attachment['data'])
                pdf_texts.append(extract_text_from_pdf_bytes(file_data))
    return pdf_texts


def extract_text_from_pdf_bytes(file_bytes):
    """Extract text from PDF byte data"""
    pdf_stream = io.BytesIO(file_bytes)
    reader = PdfReader(pdf_stream)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

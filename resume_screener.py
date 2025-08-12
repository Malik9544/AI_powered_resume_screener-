import streamlit as st
import pdfplumber
import re
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import plotly.graph_objects as go
import pandas as pd
import io

st.set_page_config(page_title="AI Resume Screener - Multi Resume Compare & Filter", page_icon="📄", layout="wide")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def split_into_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]

def get_top_matching_sentences(resume_text, jd_embedding, model, top_k=3):
    sentences = split_into_sentences(resume_text)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(jd_embedding, sentence_embeddings)[0]
    top_results = sorted(zip(scores, sentences), key=lambda x: x[0], reverse=True)[:top_k]
    return top_results

def extract_keywords(text, top_n=15):
    stop_words = set([
        "the", "and", "for", "with", "that", "this", "from", "were", "have",
        "been", "will", "are", "but", "you", "your", "not", "all", "any", "can",
        "our", "has", "had", "they", "their", "them", "which", "when", "who"
    ])
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    filtered = [w for w in words if w not in stop_words]
    most_common = Counter(filtered).most_common(top_n)
    return [word for word, _ in most_common]

def plot_pie_chart(score, key):
    fig = go.Figure(go.Pie(
        labels=['Matched', 'Unmatched'],
        values=[score, 100 - score],
        marker=dict(colors=['#4CAF50', '#E57373']),
        hole=0.4,
        hoverinfo='label+percent'
    ))
    fig.update_layout(height=300, margin=dict(t=0,b=0,l=0,r=0), showlegend=True)
    st.plotly_chart(fig, use_container_width=True, key=key)

def plot_bar_chart(top_sentences, key):
    sentences = [s if len(s) <= 80 else s[:77] + "..." for _, s in top_sentences]
    scores = [float(score) for score, _ in top_sentences]

    fig = go.Figure(go.Bar(
        x=scores[::-1],
        y=sentences[::-1],
        orientation='h',
        marker_color='#2196F3',
        hovertemplate='%{y}<br>Score: %{x:.2f}<extra></extra>'
    ))
    fig.update_layout(
        height=250,
        margin=dict(t=30,b=10,l=10,r=10),
        xaxis_title='Similarity Score',
        yaxis_title='Top Matched Sentences',
        yaxis=dict(tickfont=dict(size=12))
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

def plot_keyword_overlap(jd_keywords, resume_keywords, key):
    overlap = set(jd_keywords) & set(resume_keywords)
    only_jd = set(jd_keywords) - overlap
    only_resume = set(resume_keywords) - overlap

    labels = ['Overlap', 'Only in JD', 'Only in Resume']
    values = [len(overlap), len(only_jd), len(only_resume)]
    colors = ['#4CAF50', '#E57373', '#2196F3']

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hole=0.4,
        hoverinfo='label+percent'
    ))
    fig.update_layout(height=300, margin=dict(t=0,b=0,l=0,r=0), showlegend=True)
    st.plotly_chart(fig, use_container_width=True, key=key)

st.title("🚀 AI Resume Screener — Multi Resume Upload, Compare & Filter")

jd_text = st.text_area("Paste Job Description Text", height=180)
resume_files = st.file_uploader("Upload One or More Resumes (PDFs)", type=["pdf"], accept_multiple_files=True)

# Sidebar for criteria
st.sidebar.header("Set Screening Criteria")
min_score = st.sidebar.slider("Minimum Match Score (%)", min_value=0, max_value=100, value=70, step=1)

if jd_text.strip():
    jd_keywords = extract_keywords(jd_text, top_n=15)
    required_keywords = st.sidebar.multiselect(
        "Required Keywords (candidate must include at least one)", options=jd_keywords, default=jd_keywords[:3]
    )
else:
    required_keywords = []

if st.button("Analyze"):
    if not jd_text.strip() or not resume_files:
        st.warning("Please provide the Job Description and upload at least one Resume!")
    else:
        jd_embedding = model.encode(jd_text, convert_to_tensor=True)
        jd_keywords = extract_keywords(jd_text, top_n=15)

        results = []
        for idx, file in enumerate(resume_files):
            with st.spinner(f"Processing {file.name}..."):
                resume_text = extract_text_from_pdf(file)
                resume_embedding = model.encode(resume_text, convert_to_tensor=True)
                score = util.pytorch_cos_sim(jd_embedding, resume_embedding).item() * 100
                top_sentences = get_top_matching_sentences(resume_text, jd_embedding, model, top_k=3)
                resume_keywords = extract_keywords(resume_text, top_n=15)
                results.append({
                    'filename': file.name,
                    'score': score,
                    'top_sentences': top_sentences,
                    'resume_text': resume_text,
                    'resume_keywords': resume_keywords
                })

        # Sort by score descending
        results = sorted(results, key=lambda x: x['score'], reverse=True)

        # Function to check criteria
        def passes_criteria(res):
            if res['score'] < min_score:
                return False
            if required_keywords:
                if not set(required_keywords).intersection(set(res['resume_keywords'])):
                    return False
            return True

        filtered_results = [r for r in results if passes_criteria(r)]

        # Summary table for all candidates
        st.subheader("📋 All Candidate Match Scores")
        df_all = pd.DataFrame([
            {'Candidate': r['filename'], 'Match Score (%)': round(r['score'], 2)} for r in results
        ])
        st.dataframe(df_all.style.highlight_max(subset=['Match Score (%)'], color='lightgreen'))

        # Show filtered candidates passing criteria
        st.subheader(f"✅ Candidates Passing Filters ({len(filtered_results)} out of {len(results)})")

        if filtered_results:
            df_filtered = pd.DataFrame([
                {'Candidate': r['filename'], 'Match Score (%)': round(r['score'], 2)} for r in filtered_results
            ])
            st.dataframe(df_filtered.style.highlight_max(subset=['Match Score (%)'], color='#4CAF50'))
        else:
            st.info("No candidates match the criteria.")

        # Display detailed info for top filtered candidate
        if filtered_results:
            top_candidate = filtered_results[0]
            st.markdown(f"<h2 style='color:#4CAF50;'>Top Matched Candidate: {top_candidate['filename']} - {top_candidate['score']:.2f}%</h2>", unsafe_allow_html=True)
            plot_pie_chart(top_candidate['score'], key="top_pie")
            st.subheader("Top Matched Sentences")
            plot_bar_chart(top_candidate['top_sentences'], key="top_bar")
            for score, sentence in top_candidate['top_sentences']:
                st.markdown(f"- {sentence}  _(Score: {score:.2f})_")
            st.subheader("Keyword Overlap")
            plot_keyword_overlap(jd_keywords, top_candidate['resume_keywords'], key="top_keyword")
            st.markdown(f"**Matched Keywords:** {', '.join(set(jd_keywords) & set(top_candidate['resume_keywords'])) if set(jd_keywords) & set(top_candidate['resume_keywords']) else 'None'}")

            with st.expander(f"View Extracted Resume Text for {top_candidate['filename']} (Preview)"):
                st.text(top_candidate['resume_text'][:1000] + "...\n\n[Text truncated]")

        # Option to finalize and export shortlist
        if filtered_results:
            if st.button("Finalize & Export Shortlist as CSV"):
                csv_buffer = io.StringIO()
                export_df = pd.DataFrame([
                    {
                        'Candidate': r['filename'],
                        'Match Score (%)': round(r['score'], 2),
                        'Matched Keywords': ', '.join(set(jd_keywords) & set(r['resume_keywords']))
                    } for r in filtered_results
                ])
                export_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_buffer.getvalue(),
                    file_name="shortlisted_candidates.csv",
                    mime="text/csv"
                )
                st.success("Shortlist exported successfully!")

        # Optional: Show summaries for other filtered candidates
        if len(filtered_results) > 1:
            with st.expander("Other Finalists Summary"):
                for r in filtered_results[1:]:
                    st.markdown(f"### {r['filename']} — {r['score']:.2f}% Match")
                    plot_pie_chart(r['score'], key=f"pie_{r['filename']}")
                    st.markdown(f"**Top Keywords Matched:** {', '.join(set(jd_keywords) & set(r['resume_keywords'])) if set(jd_keywords) & set(r['resume_keywords']) else 'None'}")


import streamlit as st
from transformers import pipeline

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# --- CUSTOM UI STYLE ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #00BFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="title">📄 AI Resume Analyzer</p>', unsafe_allow_html=True)
st.write("Analyze your resume using AI + get ATS score + improvements")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload Resume (TXT only)", type=["txt"])

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")

    # --- RAG (Chunking) ---
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    # --- ATS SCORE ---
    keywords = [
        "python", "ai", "machine learning", "data",
        "analysis", "sql", "cloud", "aws", "excel"
    ]

    matched = [word for word in keywords if word in text.lower()]
    score = int((len(matched) / len(keywords)) * 100)

    # --- DISPLAY METRICS ---
    col1, col2 = st.columns(2)

    with col1:
        st.metric("📊 ATS Score", f"{score}/100")

    with col2:
        st.success(f"Matched Keywords: {', '.join(matched) if matched else 'None'}")

    # --- MODEL (BETTER OUTPUT) ---
    pipe = pipeline("text2text-generation", model="google/flan-t5-small")

    # --- SIMPLE RETRIEVAL (RAG) ---
    context = chunks[0]

    # --- PROMPT ---
    prompt = f"""
    You are a professional resume reviewer.

    Analyze the resume and provide:

    1. Strengths (bullet points)
    2. Weaknesses (bullet points)
    3. Improvements (bullet points)

    Resume:
    {context}
    """

    # --- GENERATE RESPONSE ---
    result = pipe(prompt, max_length=300)[0]["generated_text"]

    # --- OUTPUT UI ---
    st.subheader("📌 AI Suggestions")

    st.text_area("Analysis Result", result, height=300)

    # --- EXTRA SECTION ---
    st.subheader("💡 Tips to Improve ATS Score")

    st.write("""
    - Add more job-specific keywords  
    - Use clear section headings  
    - Keep resume concise (1 page)  
    - Highlight measurable achievements  
    - Use action verbs (Built, Developed, Led)
    """)

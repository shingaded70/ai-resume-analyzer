import streamlit as st
from transformers import pipeline

st.title("📄 AI Resume Analyzer")
st.write("Upload your resume and get ATS score + suggestions")

uploaded_file = st.file_uploader("Upload Resume", type=["txt"])

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")

    # Simple ATS scoring
    keywords = ["python", "ai", "machine learning", "data", "analysis"]
    score = sum(1 for word in keywords if word in text.lower()) * 20

    st.subheader(f"📊 ATS Score: {score}/100")

    # Load small fast model
    pipe = pipeline("text-generation", model="sshleifer/tiny-gpt2")

    prompt = f"""
    You are a resume expert.

    Analyze this resume and give:
    1. Strengths
    2. Weaknesses
    3. Improvements

    Resume:
    {text}
    """

    result = pipe(prompt, max_new_tokens=150)[0]["generated_text"]

    st.subheader("📌 Suggestions")
    st.write(result)

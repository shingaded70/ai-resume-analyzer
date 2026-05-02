import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

st.title("📄 AI Resume Analyzer")

uploaded_file = st.file_uploader("Upload Resume", type=["txt"])

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")

    docs = [Document(page_content=text)]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings()
    vectordb = Chroma.from_documents(splits, embeddings)

    pipe = pipeline("text-generation", model="sshleifer/tiny-gpt2")
    llm = HuggingFacePipeline(pipeline=pipe)

    score = 80  # simple ATS score

    st.write("ATS Score:", score)

    response = llm(f"Review this resume: {text}")
    st.write(response)
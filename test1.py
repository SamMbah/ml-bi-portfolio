# ------------------- Global Imports ------------------- #
import streamlit as st

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="🧠 ML Projects Portfolio", page_icon="🧠")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import uuid
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from wordcloud import WordCloud
from transformers import pipeline
from PyPDF2 import PdfReader
import docx2txt
import speech_recognition as sr
from gtts import gTTS
from deep_translator import GoogleTranslator
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import torch

# ------------------- Sidebar ------------------- #
st.sidebar.title("ML Projects Portfolio")
project = st.sidebar.radio("Choose a project:", [
    "IMDB Sentiment Analysis",
    "Malware URL Detection",
    "Loan Default Prediction",
    "Text Summarization",
    "Conversational AI"
])

# ------------------- IMDB Sentiment Analysis ------------------- #
if project == "IMDB Sentiment Analysis":
    st.title("🎬 IMDB Sentiment Analysis")

    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    df = pd.read_csv("train.csv")
    
    X_train, X_val, y_train, y_val = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)
    X_val_vec = vectorizer.transform(X_val)
    y_pred = model.predict(X_val_vec)

    tabs = st.tabs(["Prediction", "EDA", "Model Performance"])

    with tabs[0]:
        st.subheader("Prediction")
        user_input = st.text_area("Enter a movie review:")
        if st.button("Predict Sentiment"):
            input_vec = vectorizer.transform([user_input])
            prediction = model.predict(input_vec)[0]
            result = "Positive 😊" if prediction == 1 else "Negative 😞"
            st.header(f"Prediction: {result}")

    with tabs[1]:
        st.subheader("EDA")
        st.bar_chart(df["label"].value_counts())

    with tabs[2]:
        st.subheader("Model Performance")
        st.write(f"Accuracy: {accuracy_score(y_val, y_pred):.2f}")

# ------------------- Malware URL Detection ------------------- #
if project == "Malware URL Detection":
    st.title("🛡️ Malware URL Detection")
    df = pd.read_csv("malicious_phish.csv")
    df["label"] = df["type"].apply(lambda x: "Safe" if x == "benign" else "Malicious")

    if "url_model" not in st.session_state:
        X = df["url"]
        y = df["label"].map({"Safe": 0, "Malicious": 1})
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)

        st.session_state.url_model = model
        st.session_state.url_vectorizer = vectorizer
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

    model = st.session_state.url_model
    vectorizer = st.session_state.url_vectorizer
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    user_input = st.text_input("Enter a URL:")
    if st.button("Predict URL"):
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        result = "Safe ✅" if prediction == 0 else "Malicious 🚨"
        st.header(f"Prediction: {result}")

# ------------------- Loan Default Prediction ------------------- #
if project == "Loan Default Prediction":
    st.title("💳 Loan Default Prediction")
    df = pd.read_csv("clean_loan_data.csv")

    def clean_emp_length(value):
        if pd.isnull(value): return 0
        if '<' in str(value): return 0
        if '10+' in str(value): return 10
        return int(str(value).split()[0])

    df['emp_length'] = df['emp_length'].apply(clean_emp_length)
    df_model = df[['annual_inc', 'loan_amnt', 'emp_length', 'int_rate', 'target']]

    X = df_model.drop("target", axis=1)
    y = df_model["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    annual_inc = st.number_input("Annual Income (£)", min_value=0)
    loan_amnt = st.number_input("Loan Amount (£)", min_value=0)
    emp_length = st.number_input("Employment Length (Years)", min_value=0)
    int_rate = st.number_input("Interest Rate (%)", min_value=0.0)

    if st.button("Predict Loan Status"):
        input_data = np.array([[annual_inc, loan_amnt, emp_length, int_rate]])
        prediction = model.predict(input_data)[0]
        result = "Charged Off 🔥" if prediction == 1 else "Fully Paid ✅"
        st.header(f"Prediction: {result}")

# ------------------- Text Summarization ------------------- #
if project == "Text Summarization":
    st.title("📝 Text Summarization")
    
    @st.cache_resource
    def load_summarizer():
        return pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
    
    summarizer = load_summarizer()

    text_input = st.text_area("Paste text here:")
    uploaded_file = st.file_uploader("Or upload a document", type=["txt", "pdf", "docx"])

    document_text = ""
    if uploaded_file:
        if uploaded_file.type == "text/plain":
            document_text = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(uploaded_file)
            document_text = "".join(page.extract_text() for page in pdf_reader.pages)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            document_text = docx2txt.process(uploaded_file)

    final_text = text_input if text_input.strip() else document_text

    if st.button("Generate Summary") and final_text:
        with st.spinner("Summarizing..."):
            result = summarizer(final_text, max_length=150, min_length=40, do_sample=False)
            st.write(result[0]['summary_text'])

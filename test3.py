# -------------------- Global Imports -------------------- #
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
from transformers import pipeline
from PyPDF2 import PdfReader
import docx2txt
from PIL import Image
import requests

# -------------------- Helper Functions -------------------- #
def download_file_from_google_drive(url, filename):
    file_id = url.split("/d/")[1].split("/")[0]
    dwn_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    if not os.path.exists(filename):
        with open(filename, "wb") as f:
            f.write(requests.get(dwn_url).content)

# -------------------- Preload Files -------------------- #
download_file_from_google_drive("https://drive.google.com/file/d/1BX6SBPDwi1i7rSlo8kNF1zX33gEd-jP7/view?usp=sharing", "sentiment_model.pkl")
download_file_from_google_drive("https://drive.google.com/file/d/1PWLFn_5wykG6eVTLL4Qna0dCV8au7Bwd/view?usp=sharing", "tfidf_vectorizer.pkl")
download_file_from_google_drive("https://drive.google.com/file/d/1PVBRVmmd3iZpCkrTBi9jau_Z5nWhyhIL/view?usp=sharing", "train.csv")
download_file_from_google_drive("https://drive.google.com/file/d/1bRloqBiFla4qBZlVd5sXCMSIRtgAyi-O/view?usp=sharing", "malicious_phish.csv")

# -------------------- Config -------------------- #
st.set_page_config(page_title="🧠 ML & BI Portfolio", page_icon="🧠", layout="wide")
st.sidebar.title("ML & BI Portfolio")
page = st.sidebar.radio("Select Page:", ["About Me", "Machine Learning Projects", "BI Dashboards"])

# -------------------- About Me -------------------- #
if page == "About Me":
    st.title("👤 About Me")
    st.image("IMG_4202.jpg", width=200)
    st.subheader("Samuel Chukwuka Mbah")
    st.write("Data Scientist | Data Analyst | AI Developer")
    st.write("Location: Nottingham, UK | 📧 samuelmbah21@gmail.com | 📞 +44 7900361333")
    st.write("[LinkedIn](https://www.linkedin.com/in/samuel-mbah-mlengineer) | [GitHub](https://github.com/SamMbah)")

    st.write("----")
    st.subheader("Education")
    st.write("**MSc Artificial Intelligence and Data Science (Distinction)** - University of Hull (2023 - 2024)")
    st.write("**BSc Mathematics and Economics** - University of Benin, Nigeria (2011 - 2015)")

    st.write("----")
    st.subheader("Certifications")
    st.write("- Associate Data Analyst in SQL (DataCamp)")
    st.write("- Microsoft DP-203: Data Engineering on Microsoft Azure (In Progress)")
    st.write("- Python for Data Science (Coursera)")

    st.write("----")
    st.subheader("Professional Experience")
    st.write("**Customer Data Analyst / Relationship Manager - Zenith Bank Plc (2017 - 2024)**")
    st.write("Led data projects, built SQL databases, automated reporting using VBA and Power Automate.")
    st.write("**Technical Intern - Bright Network Internship Experience UK (2023)**")
    st.write("Completed training projects across Data, AI and ML.")

    st.write("----")
    st.subheader("Skills")
    st.write("""
    - **Programming**: Python, SQL, PowerShell, VBA
    - **Machine Learning**: scikit-learn, transformers, NLP, Deep Learning
    - **Data Visualization**: Power BI, Matplotlib, Seaborn, Streamlit
    - **Cloud and CI/CD**: Azure, GitHub Actions, Docker (Basic)
    - **Data Engineering**: SQL Databases, Pandas, PySpark (Basic)
    - **Generative AI**: Hugging Face Transformers, gTTS, Streamlit WebRTC
    """)

    st.write("----")
    st.subheader("Contact")
    st.write("Feel free to reach out via LinkedIn or Email for collaboration or opportunities.")

# -------------------- Machine Learning Projects -------------------- #
if page == "Machine Learning Projects":
    project = st.sidebar.radio("Choose ML Project:", [
        "IMDB Sentiment Analysis",
        "Malware URL Detection",
        "Loan Default Prediction",
        "Text Summarization"
    ])

    # ------------------- IMDB Sentiment Analysis ------------------- #
    if project == "IMDB Sentiment Analysis":
        st.title("🎜️ IMDB Sentiment Analysis")
        st.markdown("""
        This project uses Natural Language Processing to classify IMDB movie reviews as positive or negative
        using a Naive Bayes classifier trained on TF-IDF features.
        """)

        from io import BytesIO
        import requests

        def load_joblib_from_gdrive(gdrive_url):
            file_id = gdrive_url.split("/d/")[1].split("/")[0]
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            response = requests.get(download_url)
            response.raise_for_status()
            return joblib.load(BytesIO(response.content))

        model_url = "https://drive.google.com/file/d/1anBf6A7hAXiJBU53TwMsqfQmxDGRf58-/view?usp=sharing"
        vectorizer_url = "https://drive.google.com/file/d/1svkX1Lwdt8sNNHA4dtDLe1l8Lq8oewGU/view?usp=sharing"

        model = load_joblib_from_gdrive(model_url)
        vectorizer = load_joblib_from_gdrive(vectorizer_url)

        train_url = "https://drive.google.com/file/d/1PVBRVmmd3iZpCkrTBi9jau_Z5nWhyhIL/view?usp=sharing"
        train_file_id = train_url.split("/d/")[1].split("/")[0]
        train_csv_url = f"https://drive.google.com/uc?export=download&id={train_file_id}"
        df = pd.read_csv(train_csv_url)

        X_train, X_val, y_train, y_val = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)
        X_val_vec = vectorizer.transform(X_val)
        y_pred = model.predict(X_val_vec)

        tabs = st.tabs(["Prediction", "EDA", "Model Performance"])

        with tabs[0]:
            user_input = st.text_area("Enter a movie review:")
            if st.button("Predict Sentiment"):
                input_vec = vectorizer.transform([user_input])
                prediction = model.predict(input_vec)[0]
                result = "Positive 😊" if prediction == 1 else "Negative 😞"
                st.header(f"Prediction: {result}")

        with tabs[1]:
            st.bar_chart(df["label"].value_counts())

        with tabs[2]:
            st.write(f"Accuracy: {accuracy_score(y_val, y_pred):.2f}")




    # Append EDA and Model Performance Tabs to Malware URL Detection, Loan Default, and Text Summarization
    if project == "Malware URL Detection":
        st.title("🛡️ Malware URL Detection")
        st.markdown("""
        This model detects malicious URLs using Naive Bayes and TF-IDF trained on a public phishing dataset.
        Enter a URL to classify it as Safe or Malicious.
        """)
        df = pd.read_csv("malicious_phish.csv")
        df["label"] = df["type"].apply(lambda x: "Safe" if x == "benign" else "Malicious")
        X = df["url"]
        y = df["label"].map({"Safe": 0, "Malicious": 1})
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(vectorizer.transform(X_test))
        tabs = st.tabs(["Prediction", "EDA", "Model Performance"])
        with tabs[0]:
            user_input = st.text_input("Enter a URL:")
            if st.button("Predict URL"):
                input_vec = vectorizer.transform([user_input])
                prediction = model.predict(input_vec)[0]
                result = "Safe ✅" if prediction == 0 else "Malicious 🚨"
                st.header(f"Prediction: {result}")
        with tabs[1]:
            st.bar_chart(df["label"].value_counts())
        with tabs[2]:
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    if project == "Loan Default Prediction":
        st.title("💳 Loan Default Prediction")
        st.markdown("""
        Predict loan repayment outcomes based on user financial inputs. Features include income, loan amount,
        interest rate and employment length. Random Forest is used for classification.
        """)
        df = pd.read_csv("clean_loan_data.csv")
        df['emp_length'] = df['emp_length'].apply(lambda x: 0 if pd.isnull(x) or '<' in str(x) else 10 if '10+' in str(x) else int(str(x).split()[0]))
        X = df[['annual_inc', 'loan_amnt', 'emp_length', 'int_rate']]
        y = df["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        tabs = st.tabs(["Prediction", "EDA", "Model Performance"])
        with tabs[0]:
            annual_inc = st.number_input("Annual Income (£)", min_value=0)
            loan_amnt = st.number_input("Loan Amount (£)", min_value=0)
            emp_length = st.number_input("Employment Length (Years)", min_value=0)
            int_rate = st.number_input("Interest Rate (%)", min_value=0.0)
            if st.button("Predict Loan Status"):
                input_data = np.array([[annual_inc, loan_amnt, emp_length, int_rate]])
                prediction = model.predict(input_data)[0]
                result = "Charged Off 🔥" if prediction == 1 else "Fully Paid ✅"
                st.header(f"Prediction: {result}")
        with tabs[1]:
            st.bar_chart(df['target'].value_counts())
        with tabs[2]:
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    if project == "Text Summarization":
        st.title("📜 Text Summarization")
        st.markdown("""
        This tool summarizes large documents or pasted text using the Facebook BART transformer model.
        Upload .txt, .pdf, or .docx files to generate a readable summary.
        """)
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        tabs = st.tabs(["Prediction", "EDA", "Model Performance"])
        with tabs[0]:
            text_input = st.text_area("Paste text here:")
            uploaded_file = st.file_uploader("Or upload document", type=["txt", "pdf", "docx"])
            document_text = ""
            if uploaded_file:
                if uploaded_file.type == "text/plain":
                    document_text = uploaded_file.read().decode("utf-8")
                elif uploaded_file.type == "application/pdf":
                    document_text = "".join([page.extract_text() for page in PdfReader(uploaded_file).pages])
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    document_text = docx2txt.process(uploaded_file)
            final_text = text_input.strip() or document_text
            if st.button("Generate Summary") and final_text:
                result = summarizer(final_text, max_length=150, min_length=40, do_sample=False)
                st.write(result[0]['summary_text'])
        with tabs[1]:
            st.info("EDA not applicable for dynamic summarization.")
        with tabs[2]:
            st.info("Model performance metrics are not calculated live for this summarization model.")

#divide the page into two columns lines for clarification


# -------------------- BI Dashboards -------------------- #
if page == "BI Dashboards":
    dashboard = st.sidebar.radio("Choose Dashboard:", ["Healthcare Analytics", "Call Center Analytics"])

    if dashboard == "Healthcare Analytics":
        st.title("🏥 Healthcare Analytics Dashboard")
        st.markdown("""
        Explore patient demographics, hospital billing trends, and admission insights using an interactive
        Power BI dashboard with filters for year and hospital facility.
        """)

        st.image("UpdatedHealthAnalysis_page-0001.jpg", caption="Patient Overview")
        st.image("UpdatedHealthAnalysis_page-0002.jpg", caption="Hospital Impact")
        st.image("UpdatedHealthAnalysis_page-0003.jpg", caption="Admission Insight")

        with open("UpdatedHealthAnalysis.pbit", "rb") as f:
            st.download_button("Download Power BI file (.pbit)", f, file_name="HealthcareAnalytics.pbit")

    if dashboard == "Call Center Analytics":
        st.title("📞 Call Center Analytics Dashboard")
        st.markdown("""
        Visualizes call center performance including call volume, customer satisfaction, and agent metrics.
        Includes breakdowns by hour, agent, and topic.
        """)

        st.image("call center analytics.jpg", caption="Call Center Overview")

        with open("call center analytics.pbix", "rb") as f:
            st.download_button("Download Power BI file (.pbix)", f, file_name="CallCenterAnalytics.pbix")

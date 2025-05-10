# -------------------- Global Imports -------------------- #
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
from PyPDF2 import PdfReader
import docx2txt
from PIL import Image
import requests
import time
import re
from collections import Counter
import io
from io import BytesIO

# -------------------- Page Configuration -------------------- #
st.set_page_config(
    page_title="🧠 ML & BI Portfolio", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B68A2;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #4B68A2;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .project-header {
        font-size: 2.2rem;
        color: #4B68A2;
        margin-bottom: 0.8rem;
    }
    .sidebar .sidebar-content {
        background-color: #f5f7fb;
    }
    .stButton>button {
        background-color: #4B68A2;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #3A5282;
    }
    .section-divider {
        height: 3px;
        background-color: #f0f0f0;
        margin: 1.5rem 0;
    }
    .metric-card {
        background-color: #f5f7fb;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .profile-section {
        display: flex;
        align-items: center;
        margin-bottom: 1.5rem;
    }
    .profile-image {
        margin-right: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Helper Functions -------------------- #
def download_file_from_google_drive(url, filename):
    """Download a file from Google Drive given the sharing URL."""
    try:
        file_id = url.split("/d/")[1].split("/")[0]
        dwn_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        if not os.path.exists(filename):
            with open(filename, "wb") as f:
                f.write(requests.get(dwn_url).content)
            return True
        return True
    except Exception as e:
        st.error(f"Error downloading file: {e}")
        return False

def generate_wordcloud(data):
    """Generate and display a word cloud from text data."""
    wc = WordCloud(
        background_color='white', 
        max_words=100, 
        width=800, 
        height=400,
        colormap='viridis'
    ).generate(" ".join(data))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def load_joblib_from_gdrive(gdrive_url):
    """Load a joblib file directly from Google Drive."""
    file_id = gdrive_url.split("/d/")[1].split("/")[0]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(download_url)
    response.raise_for_status()
    return joblib.load(BytesIO(response.content))

# -------------------- Text Summarization Functions -------------------- #
def extract_text_from_file(uploaded_file):
    """Extract text from uploaded file (PDF or TXT)."""
    try:
        if uploaded_file.type == "text/plain":
            return uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            # Process PDF files
            pdf_bytes = io.BytesIO(uploaded_file.read())
            pdf_reader = PdfReader(pdf_bytes)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        else:
            st.error(f"Unsupported file type: {uploaded_file.type}")
            st.info("Please upload a text file (.txt) or PDF file (.pdf)")
            return ""
    except Exception as e:
        st.error(f"Error extracting text from file: {e}")
        return ""

@st.cache_data
def generate_summary(text, max_length=150, min_length=40):
    """Generate a summary using extractive summarization techniques."""
    if not text:
        return ""
    
    # Clean and preprocess text
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 3:
        return text  # Return the original text if it's already very short
    
    # Remove very short sentences
    sentences = [s for s in sentences if len(s.split()) > 3]
    
    # Calculate word frequency
    words = re.findall(r'\w+', text.lower())
    word_freq = Counter(words)
    
    # Define stopwords (common words that don't add much meaning)
    common_stopwords = {'the', 'a', 'an', 'and', 'in', 'to', 'of', 'for', 'with', 'on', 'at', 'from', 'by', 'about', 
                     'as', 'is', 'was', 'were', 'be', 'been', 'being', 'that', 'this', 'these', 'those', 'it', 'its'}
    
    # Remove stopwords from consideration
    for word in common_stopwords:
        if word in word_freq:
            del word_freq[word]
    
    # Score sentences based on word frequency
    sentence_scores = []
    for sentence in sentences:
        words_in_sentence = re.findall(r'\w+', sentence.lower())
        score = sum(word_freq.get(word, 0) for word in words_in_sentence) / len(words_in_sentence) if words_in_sentence else 0
        sentence_scores.append((sentence, score))
    
    # Sort sentences by score
    ranked_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
    
    # Select top sentences
    num_sentences = max(min(len(sentences) // 3, 10), 2)  # At least 2, at most 10 sentences
    top_sentences = ranked_sentences[:num_sentences]
    
    # Reorder sentences to maintain original order
    top_sentences = sorted(top_sentences, key=lambda x: sentences.index(x[0]))
    
    # Build summary
    summary = ' '.join(sentence for sentence, score in top_sentences)
    
    # Truncate if it's still too long
    words = summary.split()
    if len(words) > max_length:
        summary = ' '.join(words[:max_length]) + '...'
    elif len(words) < min_length and len(text.split()) > min_length:
        # Add more sentences if summary is too short
        remaining_sentences = [s for s, _ in ranked_sentences[num_sentences:num_sentences+3]]
        summary += ' ' + ' '.join(remaining_sentences)
    
    return summary

def calculate_similarity_score(summary, reference):
    """Calculate similarity between summary and reference text."""
    if not summary or not reference:
        return None
    
    try:
        # Calculate word overlap
        summary_words = set(re.findall(r'\w+', summary.lower()))
        reference_words = set(re.findall(r'\w+', reference.lower()))
        
        # Calculate metrics
        overlap = len(summary_words.intersection(reference_words))
        precision = overlap / len(summary_words) if summary_words else 0
        recall = overlap / len(reference_words) if reference_words else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'similarity': f1,
            'precision': precision, 
            'recall': recall
        }
    except Exception as e:
        st.error(f"Error calculating similarity score: {e}")
        return None

# -------------------- Preload Files -------------------- #
# Preload necessary files in a background thread to improve UX
with st.spinner("Loading resources..."):
    download_file_from_google_drive("https://drive.google.com/file/d/1BX6SBPDwi1i7rSlo8kNF1zX33gEd-jP7/view?usp=sharing", "sentiment_model.pkl")
    download_file_from_google_drive("https://drive.google.com/file/d/1PWLFn_5wykG6eVTLL4Qna0dCV8au7Bwd/view?usp=sharing", "tfidf_vectorizer.pkl")
    download_file_from_google_drive("https://drive.google.com/file/d/1PVBRVmmd3iZpCkrTBi9jau_Z5nWhyhIL/view?usp=sharing", "train.csv")
    download_file_from_google_drive("https://drive.google.com/file/d/1bRloqBiFla4qBZlVd5sXCMSIRtgAyi-O/view?usp=sharing", "malicious_phish.csv")

# -------------------- Streamlit App Layout -------------------- #
# Sidebar Configuration
st.sidebar.markdown('<h1 style="text-align: center;">ML & BI Portfolio</h1>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
page = st.sidebar.radio("Navigate:", ["About Me", "Machine Learning Projects", "BI Dashboards"])

# About Me Page
if page == "About Me":
    st.markdown('<h1 class="main-header">About Me</h1>', unsafe_allow_html=True)
    
    # Profile Section
    col1, col2 = st.columns([1, 3])
    with col1:
        original_img = Image.open("IMG_4202.jpg")
        width, height = original_img.size
        cropped_img = original_img.crop((0, 0, width, int(height * 0.6)))
        st.image(cropped_img, width=200)
    
    with col2:
        st.markdown("<h2>Samuel Chukwuka Mbah</h2>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.2rem;'>Data Scientist | Data Analyst | AI Developer</p>", unsafe_allow_html=True)
        st.write("Location: Nottingham, UK | 📧 samuelmbah21@gmail.com | 📞 +44 7900361333")
        st.write("[LinkedIn](https://www.linkedin.com/in/samuel-mbah-mlengineer) | [GitHub](https://github.com/SamMbah)")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Education Section
    st.markdown('<h2 class="sub-header">Education</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **MSc Artificial Intelligence and Data Science (Distinction)**  
        University of Hull (2023 - 2024)
        """)
    with col2:
        st.markdown("""
        **BSc Mathematics and Economics**  
        University of Benin, Nigeria (2011 - 2015)
        """)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Certifications Section
    st.markdown('<h2 class="sub-header">Certifications</h2>', unsafe_allow_html=True)
    cert_col1, cert_col2, cert_col3 = st.columns(3)
    
    with cert_col1:
        st.markdown("""
        ### Associate Data Analyst in SQL
        **DataCamp**
        """)
    
    with cert_col2:
        st.markdown("""
        ### Microsoft DP-203
        **Data Engineering on Microsoft Azure**  
        (In Progress)
        """)
    
    with cert_col3:
        st.markdown("""
        ### Python for Data Science
        **Coursera**
        """)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Professional Experience Section
    st.markdown('<h2 class="sub-header">Professional Experience</h2>', unsafe_allow_html=True)
    
    exp_col1, exp_col2 = st.columns(2)
    
    with exp_col1:
        st.markdown("""
        ### Customer Data Analyst / Relationship Manager
        **Zenith Bank Plc (2017 - 2024)**
        
        Led data projects, built SQL databases, automated reporting using VBA and Power Automate.
        """)
    
    with exp_col2:
        st.markdown("""
        ### Technical Intern
        **Bright Network Internship Experience UK (2023)**
        
        Completed training projects across Data, AI and ML.
        """)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Skills Section
    st.markdown('<h2 class="sub-header">Skills</h2>', unsafe_allow_html=True)
    
    skills_col1, skills_col2 = st.columns(2)
    
    with skills_col1:
        st.markdown("""
        - **Programming**: Python, SQL, PowerShell, VBA
        - **Machine Learning**: scikit-learn, transformers, NLP, Deep Learning
        - **Data Visualization**: Power BI, Matplotlib, Seaborn, Streamlit
        """)
    
    with skills_col2:
        st.markdown("""
        - **Cloud and CI/CD**: Azure, GitHub Actions, Docker (Basic)
        - **Data Engineering**: SQL Databases, Pandas, PySpark (Basic)
        - **Generative AI**: Hugging Face Transformers, gTTS, Streamlit WebRTC
        """)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Contact Section
    st.markdown('<h2 class="sub-header">Contact</h2>', unsafe_allow_html=True)
    st.write("Feel free to reach out via LinkedIn or Email for collaboration or opportunities.")
    
    contact_col1, contact_col2, contact_col3 = st.columns([1,1,1])
    with contact_col1:
        st.markdown("""
        📧 **Email**  
        samuelmbah21@gmail.com
        """)
    
    with contact_col2:
        st.markdown("""
        📞 **Phone**  
        +44 7900361333
        """)
    
    with contact_col3:
        st.markdown("""
        🔗 **LinkedIn**  
        [Connect with me](https://www.linkedin.com/in/samuel-mbah-mlengineer)
        """)

# -------------------- Machine Learning Projects -------------------- #
if page == "Machine Learning Projects":
    st.markdown('<h1 class="main-header">Machine Learning Projects</h1>', unsafe_allow_html=True)
    
    # Project selection
    st.sidebar.markdown('<h3>Select Project:</h3>', unsafe_allow_html=True)
    project = st.sidebar.radio("", [
        "IMDB Sentiment Analysis",
        "Malware URL Detection",
        "Loan Default Prediction",
        "Text Summarization"
    ])

    # ------------------- IMDB Sentiment Analysis ------------------- #
    if project == "IMDB Sentiment Analysis":
        st.markdown('<h2 class="project-header">🎬 IMDB Sentiment Analysis</h2>', unsafe_allow_html=True)
        st.markdown("""
        <div style='background-color:#f5f7fb; padding:1rem; border-radius:10px; margin-bottom:1rem;'>
        This project uses Natural Language Processing to classify IMDB movie reviews as positive or negative
        using a Naive Bayes classifier trained on TF-IDF features.
        </div>
        """, unsafe_allow_html=True)

        model_url = "https://drive.google.com/file/d/1anBf6A7hAXiJBU53TwMsqfQmxDGRf58-/view?usp=sharing"
        vectorizer_url = "https://drive.google.com/file/d/1svkX1Lwdt8sNNHA4dtDLe1l8Lq8oewGU/view?usp=sharing"

        with st.spinner("Loading model..."):
            model = load_joblib_from_gdrive(model_url)
            vectorizer = load_joblib_from_gdrive(vectorizer_url)

        df = pd.read_csv("train.csv")
        X_train, X_val, y_train, y_val = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)
        X_val_vec = vectorizer.transform(X_val)
        y_pred = model.predict(X_val_vec)

        tabs = st.tabs(["Prediction", "Data Analysis", "Model Performance"])

        with tabs[0]:
            st.markdown('<h3>Try the Sentiment Analyzer</h3>', unsafe_allow_html=True)
            user_input = st.text_area("Enter a movie review:", height=150)
            
            col1, col2 = st.columns([1, 3])
            with col1:
                analyze_button = st.button("Analyze Sentiment")
            
            with col2:
                if 'example_reviews' not in st.session_state:
                    st.session_state.example_reviews = [
                        "This movie was amazing! The plot was engaging and the acting was superb.",
                        "Waste of time and money. Terrible script with wooden acting throughout."
                    ]
                
                if st.button("Load Example Review"):
                    import random
                    user_input = random.choice(st.session_state.example_reviews)
                    st.session_state.review = user_input
                    st.experimental_rerun()
            
            if 'review' in st.session_state:
                user_input = st.session_state.review
            
            if analyze_button and user_input:
                with st.spinner("Analyzing sentiment..."):
                    input_vec = vectorizer.transform([user_input])
                    prediction = model.predict(input_vec)[0]
                    proba = model.predict_proba(input_vec)[0]
                    confidence = proba[1] if prediction == 1 else proba[0]
                    
                    result = "Positive 😊" if prediction == 1 else "Negative 😞"
                    
                    # Display result with styling
                    result_color = "#28a745" if prediction == 1 else "#dc3545"
                    st.markdown(f"""
                    <div style='background-color:{result_color}; padding:1rem; border-radius:10px; color:white; text-align:center; margin:1rem 0;'>
                        <h2>Prediction: {result}</h2>
                        <p>Confidence: {confidence:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)

        with tabs[1]:
            st.subheader("Class Distribution")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.countplot(x=df["label"], ax=ax, palette=["#dc3545", "#28a745"])
            ax.set_xlabel("Sentiment")
            ax.set_ylabel("Count")
            ax.set_xticklabels(["Negative", "Positive"])
            st.pyplot(fig)
            
            st.subheader("Word Cloud Visualization")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Positive Reviews")
                positive_text = df[df["label"] == 1]["text"]
                generate_wordcloud(positive_text)
            
            with col2:
                st.markdown("#### Negative Reviews")
                negative_text = df[df["label"] == 0]["text"]
                generate_wordcloud(negative_text)

        with tabs[2]:
            st.subheader("Model Performance Metrics")
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred)
            
            # Display metrics in cards
            metric1, metric2, metric3 = st.columns(3)
            metric1.metric("Accuracy", f"{accuracy:.2%}")
            metric2.metric("Training Data Size", f"{len(X_train):,}")
            metric3.metric("Validation Data Size", f"{len(X_val):,}")
            
            # Add a confusion matrix
            st.subheader("Confusion Matrix")
            from sklearn.metrics import confusion_matrix
            import seaborn as sns
            
            cm = confusion_matrix(y_val, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_xticklabels(['Negative', 'Positive'])
            ax.set_yticklabels(['Negative', 'Positive'])
            st.pyplot(fig)

    # ------------------- Malware URL Detection ------------------- #
    if project == "Malware URL Detection":
        st.markdown('<h2 class="project-header">🛡️ Malware URL Detection</h2>', unsafe_allow_html=True)
        st.markdown("""
        <div style='background-color:#f5f7fb; padding:1rem; border-radius:10px; margin-bottom:1rem;'>
        This model detects malicious URLs using Naive Bayes and TF-IDF trained on a public phishing dataset.
        Enter a URL to classify it as Safe or Malicious.
        </div>
        """, unsafe_allow_html=True)
        
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
        
        tabs = st.tabs(["URL Scanner", "Data Analysis", "Model Performance"])
        
        with tabs[0]:
            st.markdown('<h3>Check URL Safety</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                user_input = st.text_input("Enter a URL to check:", 
                                         placeholder="https://example.com")
            
            with col2:
                if st.button("Scan URL", key="scan_url_button"):
                    if not user_input:
                        st.warning("Please enter a URL")
                    else:
                        with st.spinner("Analyzing URL..."):
                            input_vec = vectorizer.transform([user_input])
                            prediction = model.predict(input_vec)[0]
                            result = "Safe ✅" if prediction == 0 else "Malicious 🚨"
                            
                            # Display result with styling
                            result_color = "#28a745" if prediction == 0 else "#dc3545"
                            st.markdown(f"""
                            <div style='background-color:{result_color}; padding:1rem; border-radius:10px; color:white; text-align:center; margin:1rem 0;'>
                                <h2>URL Status: {result}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if prediction == 1:
                                st.warning("⚠️ This URL exhibits characteristics of malicious websites. Be careful!")
                            else:
                                st.success("✅ This URL appears to be safe based on our analysis.")
        
        with tabs[1]:
            st.subheader("Dataset Overview")
            
            # Distribution pie chart
            fig, ax = plt.subplots(figsize=(8, 6))
            df["label"].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=["#28a745", "#dc3545"])
            ax.set_ylabel('')
            st.pyplot(fig)
            
            # Show sample URLs
            st.subheader("Sample URLs from Dataset")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Safe URLs (Sample)")
                safe_urls = df[df["type"] == "benign"]["url"].sample(5).reset_index(drop=True)
                st.dataframe(safe_urls)
            
            with col2:
                st.markdown("#### Malicious URLs (Sample)")
                malicious_urls = df[df["type"] != "benign"]["url"].sample(5).reset_index(drop=True)
                st.dataframe(malicious_urls)
        
        with tabs[2]:
            st.subheader("Model Performance Metrics")
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Display metrics in cards
            col1, col2 = st.columns(2)
            with col1:
                metric1, metric2 = st.columns(2)
                metric1.metric("Accuracy", f"{accuracy:.2%}")
                metric2.metric("Precision", f"{precision:.2%}")
            
            with col2:
                metric3, metric4 = st.columns(2)
                metric3.metric("Recall", f"{recall:.2%}")
                metric4.metric("F1 Score", f"{f1:.2%}")
            
            # Add confusion matrix
            st.subheader("Confusion Matrix")
            from sklearn.metrics import confusion_matrix
            
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_xticklabels(['Safe', 'Malicious'])
            ax.set_yticklabels(['Safe', 'Malicious'])
            st.pyplot(fig)

    # ------------------- Loan Default Prediction ------------------- #
    if project == "Loan Default Prediction":
        st.markdown('<h2 class="project-header">💳 Loan Default Prediction</h2>', unsafe_allow_html=True)
        st.markdown("""
        <div style='background-color:#f5f7fb; padding:1rem; border-radius:10px; margin-bottom:1rem;'>
        Predict loan repayment outcomes based on user financial inputs. Features include income, loan amount,
        interest rate and employment length. Random Forest is used for classification.
        </div>
        """, unsafe_allow_html=True)
        
        try:
            df = pd.read_csv("clean_loan_data.csv")
            df['emp_length'] = df['emp_length'].apply(lambda x: 0 if pd.isnull(x) or '<' in str(x) else 10 if '10+' in str(x) else int(str(x).split()[0]))
            X = df[['annual_inc', 'loan_amnt', 'emp_length', 'int_rate']]
            y = df["target"]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            tabs = st.tabs(["Loan Predictor", "Data Analysis", "Model Insights"])
            
            with tabs[0]:
                st.markdown('<h3>Predict Loan Default Risk</h3>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    annual_inc = st.number_input("Annual Income (£)", min_value=0, value=50000, step=5000)
                    loan_amnt = st.number_input("Loan Amount (£)", min_value=0, value=10000, step=1000)
                
                with col2:
                    emp_length = st.slider("Employment Length (Years)", min_value=0, max_value=10, value=5)
                    int_rate = st.slider("Interest Rate (%)", min_value=0.0, max_value=30.0, value=8.0, step=0.1)
                
                if st.button("Predict Loan Status", key="predict_loan_button"):
                    with st.spinner("Analyzing loan risk..."):
                        input_data = np.array([[annual_inc, loan_amnt, emp_length, int_rate]])
                        prediction = model.predict(input_data)[0]
                        probabilities = model.predict_proba(input_data)[0]
                        
                        default_prob = probabilities[1] if prediction == 1 else probabilities[0]
                        
                        result = "Charged Off 🔥" if prediction == 1 else "Fully Paid ✅"
                        
                        # Display result with styling
                        result_color = "#28a745" if prediction == 0 else "#dc3545"
                        st.markdown(f"""
                        <div style='background-color:{result_color}; padding:1rem; border-radius:10px; color:white; text-align:center; margin:1rem 0;'>
                            <h2>Prediction: {result}</h2>
                            <p>{"Default risk" if prediction == 1 else "Repayment probability"}: {max(probabilities):.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Risk factors analysis
                        st.subheader("Risk Analysis")
                        loan_to_income = loan_amnt / annual_inc
                        
                        risk_factors = []
                        if loan_to_income > 0.4:
                            risk_factors.append(f"❗ High loan-to-income ratio: {loan_to_income:.2%}")
                        if int_rate > 15:
                            risk_factors.append(f"❗ High interest rate: {int_rate}%")
                        if emp_length < 2:
                            risk_factors.append(f"❗ Short employment history: {emp_length} years")
                        
                        if risk_factors:
                            st.markdown("**Key Risk Factors:**")
                            for factor in risk_factors:
                                st.markdown(f"- {factor}")
                        else:
                            st.markdown("**No significant risk factors identified.**")
            
            with tabs[1]:
                st.subheader("Loan Data Overview")
                
                # Distribution of loan default status
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.countplot(x=df["target"], ax=ax, palette=["#28a745", "#dc3545"])
                ax.set_xlabel("Loan Status")
                ax.set_xticklabels(["Fully Paid", "Charged Off"])
                st.pyplot(fig)
                
                # Visualization of key features
                st.subheader("Feature Distributions")
                
                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.boxplot(x="target", y="annual_inc", data=df, ax=ax)
                    ax.set_xlabel("Loan Status")
                    ax.set_xticklabels(["Fully Paid", "Charged Off"])
                    ax.set_ylabel("Annual Income")
                    st.pyplot(fig)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.boxplot(x="target", y="int_rate", data=df, ax=ax)
                    ax.set_xlabel("Loan Status")
                    ax.set_xticklabels(["Fully Paid", "Charged Off"])
                    ax.set_ylabel("Interest Rate (%)")
                    st.pyplot(fig)
            
            with tabs[2]:
                st.subheader("Model Performance")
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                # Display metrics in cards
                metric1, metric2, metric3, metric4 = st.columns(4)
                metric1.metric("Accuracy", f"{accuracy:.2%}")
                metric2.metric("Precision", f"{precision:.2%}")
                metric3.metric("Recall", f"{recall:.2%}")
                metric4.metric("F1 Score", f"{f1:.2%}")
                
                # Feature importance
                st.subheader("Feature Importance")
                
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
                ax.set_title("Feature Importance in Loan Default Prediction")
                st.pyplot(fig)
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_xticklabels(['Fully Paid', 'Charged Off'])
                ax.set_yticklabels(['Fully Paid', 'Charged Off'])
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error loading loan data: {e}")
            st.info("Please make sure the clean_loan_data.csv file is uploaded to your environment.")

    # ------------------- Text Summarization ------------------- #
    if project == "Text Summarization":
        st.markdown('<h2 class="project-header">📝 Text Summarization</h2>', unsafe_allow_html=True)
        st.markdown("""
        <div style='background-color:#f5f7fb; padding:1rem; border-radius:10px; margin-bottom:1rem;'>
        This tool summarizes text using an extractive approach that identifies and extracts key sentences.
        Simply paste your text or upload a PDF/TXT file to generate a concise summary.
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs
        tabs = st.tabs(["Summarization", "Analysis", "Evaluation"])
        
        # Summarization Tab
        with tabs[0]:
            st.subheader("Generate Summary")
            
            # Text input options
            text_input = st.text_area("Paste text here:", height=200, key="summary_input")
            uploaded_file = st.file_uploader("Or upload a document", type=["txt", "pdf"], key="summary_file")
            
            # Options for summary length
            col1, col2 = st.columns(2)
            with col1:
                max_length = st.slider("Maximum summary length (words)", 50, 300, 150)
            with col2:
                min_length = st.slider("Minimum summary length (words)", 20, 100, 40)
            
            # Extract text from uploaded file
            document_text = ""
            if uploaded_file:
                with st.spinner("Extracting text from document..."):
                    document_text = extract_text_from_file(uploaded_file)
                    if document_text:
                        st.success(f"Successfully extracted {len(document_text.split())} words from the document.")
            
            # Use either text input or uploaded document
            final_text = text_input.strip() or document_text
            
            # Generate summary button
            if st.button("Generate Summary", key="generate_summary_button"):
                if not final_text:
                    st.warning("Please provide some text or upload a file.")
                else:
                    # Show progress during summarization
                    with st.spinner("Generating summary... This may take a moment."):
                        # Add a progress bar for better UX
                        progress_bar = st.progress(0)
                        for i in range(100):
                            # Simulate progress during model processing
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        # Generate the actual summary
                        summary = generate_summary(final_text, max_length=max_length, min_length=min_length)
                        
                        # Store in session state for other tabs
                        st.session_state.generated_summary = summary
                        st.session_state.original_text = final_text
                    
                    # Display the summary
                    if st.session_state.get("generated_summary", ""):
                        st.success("Summary generated successfully!")
                        st.markdown('<div style="background-color:#f5f7fb; padding:1rem; border-radius:10px; margin-top:1rem;">', unsafe_allow_html=True)
                        st.markdown("### Summary")
                        st.markdown(st.session_state.generated_summary)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Download button for the summary
                        st.download_button(
                            label="Download Summary",
                            data=st.session_state.generated_summary,
                            file_name="summary.txt",
                            mime="text/plain"
                        )
        
        # Analysis Tab
        with tabs[1]:
            st.subheader("Text Analysis")
            
            if not st.session_state.get("generated_summary", ""):
                st.info("Generate a summary first to view analysis.")
            else:
                # Display word count statistics
                original_text = st.session_state.get("original_text", "")
                summary = st.session_state.get("generated_summary", "")
                
                original_word_count = len(original_text.split())
                summary_word_count = len(summary.split())
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Original Text Words", original_word_count)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Summary Words", summary_word_count)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    compression = summary_word_count/original_word_count if original_word_count > 0 else 0
                    st.metric("Compression Ratio", f"{compression:.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Display most frequent words in original text and summary
                st.subheader("Most Frequent Words")
                
                # Create columns for original text and summary
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Text**")
                    # Get word frequency in original text
                    words = re.findall(r'\w+', original_text.lower())
                    # Filter out common stopwords
                    stopwords = {'the', 'a', 'an', 'and', 'in', 'to', 'of', 'for', 'with', 'on', 'at', 'from', 'by', 'about',
                              'as', 'is', 'was', 'were', 'be', 'been', 'being', 'that', 'this', 'these', 'those', 'it', 'its'}
                    words = [word for word in words if word not in stopwords and len(word) > 2]
                    word_freq = Counter(words).most_common(10)
                    
                    # Display as a table
                    freq_df = pd.DataFrame(word_freq, columns=["Word", "Frequency"])
                    st.dataframe(freq_df, use_container_width=True)
                
                with col2:
                    st.markdown("**Summary**")
                    # Get word frequency in summary
                    summary_words = re.findall(r'\w+', summary.lower())
                    # Filter out common stopwords
                    summary_words = [word for word in summary_words if word not in stopwords and len(word) > 2]
                    summary_word_freq = Counter(summary_words).most_common(10)
                    
                    # Display as a table
                    summary_freq_df = pd.DataFrame(summary_word_freq, columns=["Word", "Frequency"])
                    st.dataframe(summary_freq_df, use_container_width=True)
                
                # Add word cloud visualizations
                st.subheader("Word Clouds")
                
                try:
                    # Create word clouds
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Original Text**")
                        wc = WordCloud(
                            background_color='white',
                            max_words=100,
                            width=800,
                            height=400,
                            stopwords=stopwords,
                            colormap='viridis'
                        ).generate(original_text)
                        
                        fig1, ax1 = plt.subplots(figsize=(10, 5))
                        ax1.imshow(wc, interpolation='bilinear')
                        ax1.axis('off')
                        st.pyplot(fig1)
                    
                    with col2:
                        st.markdown("**Summary**")
                        summary_wc = WordCloud(
                            background_color='white',
                            max_words=50,
                            width=800,
                            height=400,
                            stopwords=stopwords,
                            colormap='plasma'
                        ).generate(summary)
                        
                        fig2, ax2 = plt.subplots(figsize=(10, 5))
                        ax2.imshow(summary_wc, interpolation='bilinear')
                        ax2.axis('off')
                        st.pyplot(fig2)
                
                except Exception as e:
                    st.error(f"Error generating word clouds: {e}")
        
        # Evaluation Tab
        with tabs[2]:
            st.subheader("Summary Evaluation")
            
            if not st.session_state.get("generated_summary", ""):
                st.info("Generate a summary first to evaluate it.")
            else:
                st.markdown("### Evaluate your summary against a reference text")
                st.markdown("Provide a reference summary (e.g., human-written) to compare with the generated summary.")
                
                reference_text = st.text_area("Paste reference text here:", key="reference_text_input")
                reference_file = st.file_uploader(
                    "Or upload reference text file", 
                    type=["txt", "pdf"], 
                    key="reference_file"
                )
                
                # Extract text from reference file if uploaded
                ref_text = ""
                if reference_file:
                    with st.spinner("Extracting text from reference document..."):
                        ref_text = extract_text_from_file(reference_file)
                
                reference_final = reference_text.strip() or ref_text
                summary = st.session_state.get("generated_summary", "")
                
                if st.button("Evaluate Summary", key="evaluate_summary_button"):
                    if not (summary and reference_final):
                        st.warning("Please generate a summary and provide a reference text.")
                    else:
                        with st.spinner("Calculating similarity scores..."):
                            scores = calculate_similarity_score(summary, reference_final)
                            
                            if scores:
                                st.success("Evaluation complete!")
                                st.markdown("### Similarity Scores")
                                st.markdown("These scores measure how similar your generated summary is to the reference text.")
                                
                                # Format scores for display
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                    st.metric("Overall Similarity", f"{scores.get('similarity', 0):.4f}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                    st.metric("Precision", f"{scores.get('precision', 0):.4f}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                with col3:
                                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                    st.metric("Recall", f"{scores.get('recall', 0):.4f}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Interpretation
                                st.markdown("#### Interpretation")
                                st.markdown("""
                                - **Overall Similarity**: Combined measure of precision and recall (F1 score)
                                - **Precision**: How many words in the generated summary are also in the reference
                                - **Recall**: How many words from the reference are captured in the generated summary
                                
                                Higher values indicate better summary quality (max 1.0).
                                """)
                            else:
                                st.error("Failed to calculate similarity scores. Please try again.")

# BI Dashboards Page
if page == "BI Dashboards":
    st.markdown('<h1 class="main-header">BI Dashboards</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color:#f5f7fb; padding:1.5rem; border-radius:10px; margin-bottom:1.5rem;'>
        <h2 style='margin-top:0'>Business Intelligence Portfolio</h2>
        <p style='font-size:1.1rem;'>
            This section showcases my Business Intelligence dashboards created with various tools including Power BI, 
            Tableau, and custom web dashboards. Each dashboard demonstrates data visualization, 
            analytical reporting, and business insights.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Coming Soon!")
    st.info("Business Intelligence dashboards are being prepared and will be available soon. Please check back later!")
    
    # Placeholder for future BI dashboards
    st.markdown("""
    #### Future Dashboards:
    
    - **Sales Performance Analytics**
    - **Financial KPI Dashboard**
    - **Marketing Campaign Effectiveness**
    - **Customer Segmentation Analysis**
    - **Supply Chain Optimization**
    """)
    
    # Display a contact form for dashboard inquiries
    st.markdown("### Interested in custom dashboards?")
    st.markdown("Feel free to reach out if you'd like to discuss custom BI solutions for your business needs.")
    
    contact_col1, contact_col2 = st.columns(2)
    with contact_col1:
        st.text_input("Name", placeholder="Your name")
        st.text_input("Email", placeholder="Your email")
    
    with contact_col2:
        st.text_area("Message", placeholder="Your inquiry about BI solutions", height=123)
    
    st.button("Send Inquiry (Coming Soon)")

# Main application execution
try:
    # Add footer
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; color:#888; padding:1rem;'>
        © 2025 Samuel Chukwuka Mbah | Portfolio created with Streamlit | Last updated: May 2025
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.markdown("Try reloading the page or contact the developer for assistance.")
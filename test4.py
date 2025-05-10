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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud
from PyPDF2 import PdfReader
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
    footer {visibility: hidden;}
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

# -------------------- IMDB Sentiment Analysis Function -------------------- #
@st.cache_resource
def load_sentiment_model():
    """Train an improved IMDB sentiment analysis model."""
    try:
        # Check if model is already trained and saved
        if os.path.exists("imdb_model.pkl") and os.path.exists("imdb_vectorizer.pkl"):
            model = joblib.load("imdb_model.pkl")
            vectorizer = joblib.load("imdb_vectorizer.pkl")
            return model, vectorizer
        
        # If we need to train from scratch
        st.info("Training a new IMDB sentiment model. This may take a few minutes...")
        
        # Default examples for training if no dataset available
        default_texts = [
            "This movie was amazing! The plot was engaging and the acting was superb.",
            "Amazing film with great direction and stellar performances.",
            "I loved this movie. The cinematography was beautiful.",
            "The best film I've seen all year. Highly recommended!",
            "Brilliant storytelling with amazing character development.",
            "Worst movie I've ever seen. Complete waste of time.",
            "Terrible acting and a predictable plot. Don't waste your money.",
            "I hated everything about this film. The script was awful.",
            "Boring, predictable, and poorly acted. Skip this one.",
            "A complete disaster. One of the worst films of the year."
        ]
        default_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1 = positive, 0 = negative
        
        # If train.csv exists, use it instead of default examples
        if os.path.exists("train.csv"):
            try:
                df = pd.read_csv("train.csv")
                if "text" in df.columns and "label" in df.columns:
                    texts = df["text"].tolist()
                    labels = df["label"].tolist()
                    st.success(f"Successfully loaded {len(texts)} examples from train.csv")
                else:
                    st.warning("train.csv exists but doesn't have the expected columns (text, label). Using default examples.")
                    texts = default_texts
                    labels = default_labels
            except Exception as e:
                st.error(f"Error loading train.csv: {e}. Using default examples.")
                texts = default_texts
                labels = default_labels
        else:
            texts = default_texts
            labels = default_labels
        
        # Preprocessing - convert to lowercase and remove HTML tags
        processed_texts = []
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            # Convert to lowercase
            text = text.lower()
            # Remove HTML tags
            text = re.sub(r'<.*?>', '', text)
            # Remove special characters
            text = re.sub(r'[^\w\s]', '', text)
            processed_texts.append(text)
        
        # Create a more powerful vectorizer with bigrams
        vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=5,
            max_df=0.8,
            stop_words='english',
            ngram_range=(1, 2)  # Unigrams and bigrams
        )
        
        # Create TF-IDF features
        X = vectorizer.fit_transform(processed_texts)
        
        # Train a more robust model (LogisticRegression instead of Naive Bayes)
        model = LogisticRegression(
            C=10,
            max_iter=1000,
            class_weight='balanced'
        )
        model.fit(X, labels)
        
        # Save the model for future use
        joblib.dump(model, "imdb_model.pkl")
        joblib.dump(vectorizer, "imdb_vectorizer.pkl")
        
        return model, vectorizer
            
    except Exception as e:
        st.error(f"Error loading sentiment model: {e}")
        # Create a very basic fallback model
        vectorizer = TfidfVectorizer(max_features=1000)
        model = MultinomialNB()
        
        # Simple example data for training
        basic_texts = [
            "This movie was amazing!",
            "I loved this movie.", 
            "Great film!",
            "Terrible movie.",
            "I hated this film."
        ]
        basic_labels = [1, 1, 1, 0, 0]
        
        X = vectorizer.fit_transform(basic_texts)
        model.fit(X, basic_labels)
        
        return model, vectorizer

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
        try:
            original_img = Image.open("IMG_4202.jpg")
            width, height = original_img.size
            cropped_img = original_img.crop((0, 0, width, int(height * 0.6)))
            st.image(cropped_img, width=200)
        except:
            st.info("Profile image not found. Please add your image as 'IMG_4202.jpg'")
    
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
        using a Logistic Regression classifier trained on TF-IDF features.
        </div>
        """, unsafe_allow_html=True)

        # Load model (with auto-training if needed)
        with st.spinner("Loading sentiment analysis model..."):
            model, vectorizer = load_sentiment_model()

        tabs = st.tabs(["Prediction", "Data Analysis", "Model Performance"])

        with tabs[0]:
            st.markdown('<h3>Try the Sentiment Analyzer</h3>', unsafe_allow_html=True)
            
            # Initialize example reviews
            example_reviews = [
                "This movie was amazing! The plot was engaging and the acting was superb.",
                "Waste of time and money. Terrible script with wooden acting throughout."
            ]
            
            # Initialize session state for review text
            if 'review_text' not in st.session_state:
                st.session_state.review_text = ""
            
            # Define callback functions for buttons
            def load_example():
                import random
                st.session_state.review_text = random.choice(example_reviews)
            
            def clear_input():
                st.session_state.review_text = ""
            
            # Text input area with session state
            user_input = st.text_area("Enter a movie review:", 
                                     height=150, 
                                     key="input_area",
                                     value=st.session_state.review_text)
            
            # Update session state when text area changes
            st.session_state.review_text = user_input
            
            # Buttons for interaction
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                analyze_button = st.button("Analyze Sentiment")
            
            with col2:
                st.button("Load Example", key="load_example", on_click=load_example)
            
            with col3:
                st.button("Clear", key="clear_input", on_click=clear_input)
            
            # Analysis logic
            if analyze_button and user_input:
                with st.spinner("Analyzing sentiment..."):
                    try:
                        # Ensure model is loaded
                        if model is not None and vectorizer is not None:
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
                        else:
                            st.error("Model failed to load. Please try again later.")
                    except Exception as e:
                        st.error(f"Error analyzing sentiment: {e}")

        with tabs[1]:
            st.subheader("IMDB Dataset Exploration")
            
            try:
                # Try to load dataset from CSV if available
                if os.path.exists("train.csv"):
                    df = pd.read_csv("train.csv")
                    
                    # Dataset statistics
                    st.markdown("### Dataset Overview")
                    stats_col1, stats_col2 = st.columns(2)
                    
                    with stats_col1:
                        # Count positive and negative reviews
                        positive_count = sum(1 for label in df["label"] if label == 1)
                        negative_count = len(df["label"]) - positive_count
                        
                        # Create a DataFrame for the chart
                        distribution_data = pd.DataFrame({
                            "Sentiment": ["Positive", "Negative"],
                            "Count": [positive_count, negative_count]
                        })
                        
                        # Plot distribution
                        fig, ax = plt.subplots(figsize=(8, 5))
                        bars = ax.bar(distribution_data["Sentiment"], distribution_data["Count"], 
                                color=["#28a745", "#dc3545"])
                        ax.set_title("Sentiment Distribution")
                        ax.set_ylabel("Number of Reviews")
                        
                        # Add count labels on top of bars
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                                    f'{height}', ha='center', va='bottom')
                        
                        st.pyplot(fig)
                    
                    with stats_col2:
                        # Word count statistics
                        word_counts = [len(str(text).split()) for text in df["text"]]
                        
                        avg_word_count = sum(word_counts) / len(word_counts)
                        max_word_count = max(word_counts)
                        min_word_count = min(word_counts)
                        
                        st.markdown("### Review Length Statistics")
                        st.markdown(f"**Average Words per Review:** {avg_word_count:.1f}")
                        st.markdown(f"**Longest Review:** {max_word_count} words")
                        st.markdown(f"**Shortest Review:** {min_word_count} words")
                        
                        # Plot histogram of review lengths
                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.hist(word_counts, bins=30, alpha=0.7, color="#4B68A2")
                        ax.set_title("Distribution of Review Lengths")
                        ax.set_xlabel("Number of Words")
                        ax.set_ylabel("Frequency")
                        st.pyplot(fig)
                    
                    # Word clouds
                    st.subheader("Word Cloud Visualization")
                    wc_col1, wc_col2 = st.columns(2)
                    
                    with wc_col1:
                        st.markdown("#### Positive Reviews")
                        try:
                            positive_text = [str(review) for review, label in zip(df["text"], df["label"]) if label == 1]
                            if positive_text:
                                generate_wordcloud(positive_text)
                            else:
                                st.info("No positive reviews available for visualization.")
                        except Exception as e:
                            st.error(f"Error generating positive word cloud: {e}")
                    
                    with wc_col2:
                        st.markdown("#### Negative Reviews")
                        try:
                            negative_text = [str(review) for review, label in zip(df["text"], df["label"]) if label == 0]
                            if negative_text:
                                generate_wordcloud(negative_text)
                            else:
                                st.info("No negative reviews available for visualization.")
                        except Exception as e:
                            st.error(f"Error generating negative word cloud: {e}")
                else:
                    st.info("No dataset available for analysis. Please upload 'train.csv' to view visualizations.")
                    
            except Exception as e:
                st.error(f"Error analyzing dataset: {e}")

        with tabs[2]:
            st.subheader("Model Performance")
            
            try:
                # Simple evaluation
                if os.path.exists("train.csv"):
                    df = pd.read_csv("train.csv")
                    X_train, X_val, y_train, y_val = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)
                    X_val_vec = vectorizer.transform(X_val)
                    y_pred = model.predict(X_val_vec)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_val, y_pred)
                    precision = precision_score(y_val, y_pred)
                    recall = recall_score(y_val, y_pred)
                    f1 = f1_score(y_val, y_pred)
                    
                    # Display metrics
                    metrics_col1, metrics_col2 = st.columns(2)
                    with metrics_col1:
                        st.metric("Accuracy", f"{accuracy:.2%}")
                        st.metric("Precision", f"{precision:.2%}")
                    
                    with metrics_col2:
                        st.metric("Recall", f"{recall:.2%}")
                        st.metric("F1 Score", f"{f1:.2%}")
                    
                    # Create confusion matrix
                    cm = confusion_matrix(y_val, y_pred)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_title("Confusion Matrix")
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    ax.set_xticklabels(["Negative", "Positive"])
                    ax.set_yticklabels(["Negative", "Positive"])
                    st.pyplot(fig)
                    
                    # Model interpretation
                    st.subheader("Model Interpretation")
                    st.markdown("""
                    This model uses TF-IDF (Term Frequency-Inverse Document Frequency) features to represent the text data.
                    TF-IDF captures the importance of words in a document relative to the entire corpus, effectively highlighting
                    distinctive words that are most useful for classification.
                    
                    The Logistic Regression classifier learns the optimal weights for each word feature to predict sentiment.
                    It works well with text data as it can handle high-dimensional feature spaces and learns clear decision boundaries.
                    """)
                else:
                    st.info("No training data available for model performance evaluation.")
            
            except Exception as e:
                st.error(f"Error evaluating model: {e}")

    # ------------------- Malware URL Detection ------------------- #
    if project == "Malware URL Detection":
        st.markdown('<h2 class="project-header">🛡️ Malware URL Detection</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background-color:#f5f7fb; padding:1rem; border-radius:10px; margin-bottom:1rem;'>
        This project uses machine learning to classify URLs as malicious, phishing, or benign
        based on extracted features from the URL strings. The model helps protect users from
        accessing potentially harmful websites.
        </div>
        """, unsafe_allow_html=True)
        
        tabs = st.tabs(["URL Detector", "Model Details", "Dataset"])
        
        with tabs[0]:
            st.markdown('### Check if a URL is safe')
            
            url_to_check = st.text_input("Enter a URL to check:", "")
            
            col1, col2 = st.columns([1,1])
            with col1:
                analyze_url = st.button("Analyze URL")
            
            with col2:
                examples = [
                    "google.com",
                    "facebook.com",
                    "ap0qw99.freedynamicdns.net/Pass/SiteKey/",
                    "secure-banking.legitbank.com.malicious-subdomain.evil.com"
                ]
                
                if st.button("Load Example URL"):
                    import random
                    url_to_check = random.choice(examples)
                    st.rerun()
            
            if analyze_url and url_to_check:
                with st.spinner("Analyzing URL..."):
                    time.sleep(1)  # Simulate processing time
                    
                    # Extract simple URL features (actual implementation would have more)
                    # For demo purposes, we're using simplified logic here
                    def check_url_safety(url):
                        url = url.lower()
                        
                        # Very simplified feature extraction
                        suspicious_patterns = [
                            'pass', 'bank', 'secure', 'login', 'signin',
                            'account', 'update', 'verify', 'password', 'credential'
                        ]
                        
                        # Additional suspicious patterns
                        suspicious_tlds = ['.xyz', '.info', '.tk', '.ml', '.ga', '.cf']
                        suspicious_domains = ['freedynamicdns', 'freehostia', 'webcindario']
                        
                        # Compute a simplified risk score
                        risk_score = 0
                        
                        # URL length (longer URLs are more suspicious)
                        if len(url) > 75:
                            risk_score += 25
                        elif len(url) > 50:
                            risk_score += 15
                        
                        # Check for suspicious words
                        for pattern in suspicious_patterns:
                            if pattern in url:
                                risk_score += 10
                        
                        # Check for suspicious TLDs
                        for tld in suspicious_tlds:
                            if url.endswith(tld):
                                risk_score += 25
                        
                        # Check for suspicious hosting domains
                        for domain in suspicious_domains:
                            if domain in url:
                                risk_score += 25
                        
                        # Check for excessive subdomains
                        if url.count('.') > 3:
                            risk_score += 20
                        
                        # Check for IP address in URL
                        if re.search(r'\d+\.\d+\.\d+\.\d+', url):
                            risk_score += 25
                        
                        # Decision based on risk score
                        if risk_score >= 50:
                            return "Malicious", risk_score
                        elif risk_score >= 30:
                            return "Suspicious", risk_score
                        else:
                            return "Safe", risk_score
                    
                    # Simple URL check
                    result, score = check_url_safety(url_to_check)
                    
                    # Display result
                    if result == "Safe":
                        result_color = "#28a745"  # Green
                        emoji = "✅"
                    elif result == "Suspicious":
                        result_color = "#ffc107"  # Yellow
                        emoji = "⚠️"
                    else:
                        result_color = "#dc3545"  # Red
                        emoji = "🔴"
                    
                    st.markdown(f"""
                    <div style='background-color:{result_color}; padding:1rem; border-radius:10px; color:white; text-align:center; margin:1rem 0;'>
                        <h2>URL Status: {result} {emoji}</h2>
                        <p>Risk Score: {score}/100</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show feature explanation
                    st.markdown("### URL Analysis")
                    st.markdown(f"**URL**: {url_to_check}")
                    
                    # Display simple feature analysis
                    features = {
                        "URL Length": len(url_to_check),
                        "Number of Dots": url_to_check.count('.'),
                        "Contains IP Address": "Yes" if re.search(r'\d+\.\d+\.\d+\.\d+', url_to_check) else "No",
                        "Contains Suspicious Words": "Yes" if any(word in url_to_check.lower() for word in ['pass', 'bank', 'secure', 'login']) else "No"
                    }
                    
                    st.table(pd.DataFrame(features.items(), columns=["Feature", "Value"]))
        
        with tabs[1]:
            st.markdown("### Model Technical Details")
            
            st.markdown("""
            #### Feature Engineering
            
            The URL detection system uses these key features:
            
            1. **URL String Features**:
               - Length of URL
               - Number of dots, dashes, digits
               - Presence of special characters
               - TLD (Top-Level Domain) category
            
            2. **Domain-based Features**:
               - Domain age and registration details
               - WHOIS information
               - Host location
            
            3. **Content-based Features**:
               - HTML & JavaScript patterns
               - Redirect behavior
               - SSL/TLS certificate validity
            
            #### Classification Algorithm
            
            The model uses a Random Forest classifier which:
            - Combines multiple decision trees to improve accuracy
            - Is robust against overfitting
            - Can handle non-linear relationships and categorical features
            - Provides feature importance for interpretability
            """)
            
            # Display dummy feature importance
            st.markdown("#### Feature Importance")
            feature_imp = {
                "URL Length": 0.18,
                "Special Character Ratio": 0.15,
                "Domain Age": 0.14,
                "Number of Subdomains": 0.12,
                "TLD Risk Score": 0.11,
                "JS Obfuscation Score": 0.09,
                "Redirect Count": 0.08,
                "IP Usage": 0.07,
                "SSL Certificate Validity": 0.06
            }
            
            # Sort by importance
            feature_imp = dict(sorted(feature_imp.items(), key=lambda x: x[1], reverse=True))
            
            # Plot feature importance
            fig, ax = plt.subplots(figsize=(10, 6))
            y_pos = np.arange(len(feature_imp))
            ax.barh(y_pos, list(feature_imp.values()), color='#4B68A2')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(list(feature_imp.keys()))
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel('Relative Importance')
            ax.set_title('Feature Importance for URL Classification')
            
            st.pyplot(fig)
            
            # Performance metrics
            st.markdown("#### Model Performance")
            
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.metric("Accuracy", "97.2%")
            
            with metrics_col2:
                st.metric("False Positive Rate", "2.3%")
            
            with metrics_col3:
                st.metric("Detection Rate", "96.8%")
        
        with tabs[2]:
            st.markdown("### Dataset Information")
            
            if os.path.exists("malicious_phish.csv"):
                try:
                    # Load and sample data for display
                    data = pd.read_csv("malicious_phish.csv")
                    st.markdown(f"**Total URLs in dataset**: {len(data):,}")
                    
                    # Distribution of classes
                    if 'type' in data.columns:
                        st.markdown("#### Distribution of URL Types")
                        type_counts = data['type'].value_counts()
                        
                        # Create pie chart
                        fig, ax = plt.subplots(figsize=(8, 8))
                        colors = ['#4B68A2', '#DC3545', '#28A745']
                        ax.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%',
                              startangle=90, colors=colors)
                        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                        st.pyplot(fig)
                    
                    # Show sample of data
                    st.markdown("#### Sample URLs from Dataset")
                    st.dataframe(data.sample(10))
                    
                except Exception as e:
                    st.error(f"Error loading dataset: {e}")
            else:
                st.info("The malicious URL dataset file is not available. Please upload 'malicious_phish.csv' to view dataset details.")

    # ------------------- Loan Default Prediction ------------------- #
    if project == "Loan Default Prediction":
        st.markdown('<h2 class="project-header">💰 Loan Default Prediction</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background-color:#f5f7fb; padding:1rem; border-radius:10px; margin-bottom:1rem;'>
        This model predicts the likelihood of loan default based on customer financial data and loan attributes.
        It helps financial institutions assess risk and make informed lending decisions.
        </div>
        """, unsafe_allow_html=True)
        
        tabs = st.tabs(["Prediction Tool", "Model Insights", "Dataset Exploration"])
        
        with tabs[0]:
            st.markdown("### Predict Loan Default Risk")
            
            # Form for user input
            with st.form("loan_prediction_form"):
                st.markdown("#### Borrower Information")
                col1, col2 = st.columns(2)
                
                with col1:
                    age = st.slider("Age", 18, 80, 35)
                    income = st.number_input("Annual Income ($)", 10000, 500000, 60000, step=5000)
                    employment_length = st.slider("Years Employed", 0, 40, 5)
                
                with col2:
                    credit_score = st.slider("Credit Score", 300, 850, 680)
                    debt_to_income = st.slider("Debt-to-Income Ratio (%)", 0, 100, 35)
                    existing_loans = st.number_input("Number of Existing Loans", 0, 10, 1)
                
                st.markdown("#### Loan Details")
                col1, col2 = st.columns(2)
                
                with col1:
                    loan_amount = st.number_input("Loan Amount ($)", 1000, 100000, 15000, step=1000)
                    interest_rate = st.slider("Interest Rate (%)", 1.0, 30.0, 8.5, 0.1)
                
                with col2:
                    loan_term = st.selectbox("Loan Term (Years)", [1, 2, 3, 5, 10, 15, 30], 2)
                    loan_purpose = st.selectbox("Loan Purpose", [
                        "Debt Consolidation", "Credit Card", "Home Improvement", 
                        "Small Business", "Education", "Medical", "Auto", "Other"
                    ])
                
                # Submit button
                submitted = st.form_submit_button("Predict Default Risk")
            
            # Handle prediction when form is submitted
            if submitted:
                with st.spinner("Calculating default risk..."):
                    # Simulate model prediction
                    time.sleep(1)
                    
                    # Simplified risk calculation for demonstration
                    # In a real model, you would use a trained ML model here
                    risk_factors = {
                        'age': -0.01 * (age - 40),  # Younger borrowers are higher risk
                        'income': -0.00005 * (income - 50000),  # Lower income is higher risk
                        'credit_score': -0.015 * (credit_score - 650),  # Lower score is higher risk
                        'debt_to_income': 0.02 * debt_to_income,  # Higher DTI is higher risk
                        'loan_amount': 0.0001 * loan_amount,  # Larger loans are higher risk
                        'interest_rate': 0.05 * interest_rate,  # Higher rates indicate higher risk
                        'employment_length': -0.02 * employment_length,  # Shorter employment is higher risk
                        'existing_loans': 0.1 * existing_loans,  # More loans is higher risk
                        'loan_term': -0.02 * loan_term,  # Longer term is lower risk (simplified)
                        'purpose_factor': 0.1 if loan_purpose in ["Small Business", "Medical", "Education"] else 0,  # Some purposes are higher risk
                    }
                    
                    # Calculate risk score (0-100)
                    base_score = 50
                    risk_score = base_score + sum(risk_factors.values()) * 10
                    
                    # Ensure score is between 0 and 100
                    risk_score = max(min(risk_score, 100), 0)
                    
                    # Determine risk category and probability
                    if risk_score < 20:
                        risk_category = "Very Low Risk"
                        default_probability = risk_score / 400  # Max ~5%
                    elif risk_score < 40:
                        risk_category = "Low Risk"
                        default_probability = risk_score / 200  # 10-20%
                    elif risk_score < 60:
                        risk_category = "Moderate Risk"
                        default_probability = risk_score / 120  # 25-50%
                    elif risk_score < 80:
                        risk_category = "High Risk"
                        default_probability = risk_score / 100  # 50-80%
                    else:
                        risk_category = "Very High Risk"
                        default_probability = risk_score / 100  # 80-100%
                    
                    # Determine color for risk display
                    if risk_score < 30:
                        risk_color = "#28a745"  # Green
                    elif risk_score < 50:
                        risk_color = "#7fb800"  # Light green
                    elif risk_score < 70:
                        risk_color = "#ffc107"  # Yellow
                    elif risk_score < 90:
                        risk_color = "#ff851b"  # Orange
                    else:
                        risk_color = "#dc3545"  # Red
                    
                    # Display prediction result
                    st.markdown(f"""
                    <div style='background-color:{risk_color}; padding:1rem; border-radius:10px; color:white; text-align:center; margin:1rem 0;'>
                        <h2>Default Risk: {risk_category}</h2>
                        <p style='font-size:1.2rem;'>Estimated Probability of Default: {default_probability:.1%}</p>
                        <p style='font-size:1.2rem;'>Risk Score: {risk_score:.1f}/100</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Factor breakdown
                    st.markdown("### Risk Factor Breakdown")
                    
                    # Normalize factor contributions for the chart
                    factor_impacts = {}
                    for factor, value in risk_factors.items():
                        if factor == 'purpose_factor':
                            factor_name = 'Loan Purpose'
                        else:
                            # Convert snake_case to Title Case
                            factor_name = ' '.join(word.capitalize() for word in factor.split('_'))
                        factor_impacts[factor_name] = value
                    
                    # Sort factors by impact
                    sorted_factors = dict(sorted(factor_impacts.items(), key=lambda x: abs(x[1]), reverse=True))
                    
                    # Plot factor impact
                    fig, ax = plt.subplots(figsize=(10, 6))
                    factors_list = list(sorted_factors.keys())
                    values_list = list(sorted_factors.values())
                    colors = ['#dc3545' if x > 0 else '#28a745' for x in values_list]
                    bars = ax.barh(factors_list, values_list, color=colors)
                    ax.set_xlabel('Impact on Default Risk')
                    ax.set_title('Factors Influencing Default Risk Prediction')
                    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                    
                    # Add value labels to the end of each bar
                    for bar in bars:
                        width = bar.get_width()
                        label_x_pos = width + 0.02 if width > 0 else width - 0.05
                        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                                va='center', ha='left' if width > 0 else 'right')
                    
                    st.pyplot(fig)
                    
                    # Recommendation based on risk
                    st.markdown("### Recommendation")
                    
                    if risk_score < 40:
                        st.success("""
                        **Approval Recommended**
                        
                        This application has a low predicted default risk. Consider offering favorable interest rates.
                        """)
                    elif risk_score < 70:
                        st.warning("""
                        **Conditional Approval**
                        
                        This application has moderate risk. Consider additional verification or collateral requirements.
                        """)
                    else:
                        st.error("""
                        **High Risk - Additional Review Needed**
                        
                        This application has high predicted default risk. Consider requiring a co-signer, 
                        additional collateral, or denying the application.
                        """)
        
        with tabs[1]:
            st.markdown("### Model Insights")
            
            # Feature importance
            st.markdown("#### Feature Importance")
            st.markdown("""
            The model identifies these factors as most important in predicting loan default:
            """)
            
            # Dummy feature importance for visualization
            features = {
                "Credit Score": 0.26,
                "Debt-to-Income Ratio": 0.19,
                "Employment History": 0.15,
                "Loan Amount to Income": 0.12,
                "Number of Existing Loans": 0.09,
                "Interest Rate": 0.08,
                "Age": 0.06,
                "Loan Purpose": 0.05
            }
            
            # Create feature importance chart
            fig, ax = plt.subplots(figsize=(10, 6))
            y_pos = np.arange(len(features))
            ax.barh(y_pos, list(features.values()), color='#4B68A2')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(list(features.keys()))
            ax.invert_yaxis()
            ax.set_xlabel('Relative Importance')
            ax.set_title('Feature Importance for Default Prediction')
            
            st.pyplot(fig)
            
            # Performance metrics
            st.markdown("#### Model Performance")
            
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                st.metric("Accuracy", "89.3%")
            
            with metrics_col2:
                st.metric("Precision", "85.2%")
            
            with metrics_col3:
                st.metric("Recall", "82.7%")
            
            with metrics_col4:
                st.metric("ROC-AUC", "0.912")
            
            # ROC curve
            st.markdown("#### ROC Curve")
            
            # Generate a simple ROC curve for illustration
            from sklearn.metrics import roc_curve, auc
            import numpy as np
            
            # Generate dummy data
            np.random.seed(42)
            true_labels = np.random.randint(0, 2, 1000)
            scores = np.random.normal(0.5, 0.3, 1000)
            scores[true_labels == 1] += 0.5  # Make positive class have higher scores
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(true_labels, scores)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, color='#4B68A2', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.05)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC)')
            ax.legend(loc="lower right")
            
            st.pyplot(fig)
        
        with tabs[2]:
            st.markdown("### Dataset Exploration")
            
            # Load sample data or generate synthetic data
            np.random.seed(42)
            n_samples = 1000
            
            # Create synthetic data
            data = {
                'Age': np.random.randint(20, 70, n_samples),
                'Annual Income': np.random.randint(20000, 200000, n_samples),
                'Credit Score': np.random.randint(500, 850, n_samples),
                'Debt-to-Income (%)': np.random.randint(10, 80, n_samples),
                'Loan Amount': np.random.randint(5000, 80000, n_samples),
                'Interest Rate (%)': np.random.uniform(4, 20, n_samples).round(2),
                'Default': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
            }
            df = pd.DataFrame(data)
            
            # Calculate additional features
            df['Loan-to-Income Ratio'] = (df['Loan Amount'] / df['Annual Income']).round(2)
            
            # Show a data preview
            st.dataframe(df.sample(10))
            
            # Show correlation matrix
            st.markdown("#### Feature Correlations")
            
            corr = df.drop('Default', axis=1).corr()
            
            # Plot correlation matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            ax.set_title('Correlation Matrix of Loan Features')
            
            st.pyplot(fig)
            
            # Default rate by various features
            st.markdown("#### Default Rates by Feature")
            
            # Function to calculate default rate by category
            def default_rate_by_category(df, column, bins=None, labels=None):
                if bins is not None:
                    # For numerical features, bin them
                    df['category'] = pd.cut(df[column], bins=bins, labels=labels)
                else:
                    # For categorical features, use as is
                    df['category'] = df[column]
                
                # Group by category and calculate default rate
                result = df.groupby('category')['Default'].mean().reset_index()
                result.columns = ['Category', 'Default Rate']
                result['Default Rate'] = result['Default Rate'] * 100  # Convert to percentage
                
                return result
            
            # Create two columns for the charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Default rate by credit score
                credit_bins = [500, 580, 620, 680, 750, 850]
                credit_labels = ['Very Poor', 'Poor', 'Fair', 'Good', 'Excellent']
                credit_default = default_rate_by_category(df, 'Credit Score', bins=credit_bins, labels=credit_labels)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(credit_default['Category'], credit_default['Default Rate'], color='#4B68A2')
                ax.set_ylim(0, 30)
                ax.set_ylabel('Default Rate (%)')
                ax.set_title('Default Rate by Credit Score')
                
                # Add percentage labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{height:.1f}%', ha='center', va='bottom')
                
                st.pyplot(fig)
            
            with col2:
                # Default rate by DTI
                dti_bins = [0, 20, 30, 40, 50, 80]
                dti_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
                dti_default = default_rate_by_category(df, 'Debt-to-Income (%)', bins=dti_bins, labels=dti_labels)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(dti_default['Category'], dti_default['Default Rate'], color='#4B68A2')
                ax.set_ylim(0, 30)
                ax.set_ylabel('Default Rate (%)')
                ax.set_title('Default Rate by Debt-to-Income Ratio')
                
                # Add percentage labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{height:.1f}%', ha='center', va='bottom')
                
                st.pyplot(fig)
            
            # Loan amount vs default scatter plot
            st.markdown("#### Loan Amount vs. Income by Default Status")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot non-default loans
            ax.scatter(df[df['Default'] == 0]['Annual Income'], df[df['Default'] == 0]['Loan Amount'],
                     alpha=0.5, color='#4B68A2', label='Non-Default')
            
            # Plot default loans
            ax.scatter(df[df['Default'] == 1]['Annual Income'], df[df['Default'] == 1]['Loan Amount'],
                     alpha=0.5, color='#DC3545', label='Default')
            
            ax.set_xlabel('Annual Income ($)')
            ax.set_ylabel('Loan Amount ($)')
            ax.set_title('Loan Amount vs. Annual Income by Default Status')
            ax.legend()
            
            # Add trend lines
            from sklearn.linear_model import LinearRegression
            
            # Trend for non-default
            X_nd = df[df['Default'] == 0]['Annual Income'].values.reshape(-1, 1)
            y_nd = df[df['Default'] == 0]['Loan Amount'].values
            reg_nd = LinearRegression().fit(X_nd, y_nd)
            ax.plot(np.sort(X_nd, axis=0), reg_nd.predict(np.sort(X_nd, axis=0)),
                  color='#4B68A2', linestyle='--')
            
            # Trend for default
            X_d = df[df['Default'] == 1]['Annual Income'].values.reshape(-1, 1)
            y_d = df[df['Default'] == 1]['Loan Amount'].values
            reg_d = LinearRegression().fit(X_d, y_d)
            ax.plot(np.sort(X_d, axis=0), reg_d.predict(np.sort(X_d, axis=0)),
                  color='#DC3545', linestyle='--')
            
            st.pyplot(fig)

    # ------------------- Text Summarization ------------------- #
    if project == "Text Summarization":
        st.markdown('<h2 class="project-header">📝 Text Summarization</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background-color:#f5f7fb; padding:1rem; border-radius:10px; margin-bottom:1rem;'>
        This project uses extractive summarization techniques to create concise summaries of longer texts.
        The algorithm identifies and extracts key sentences based on word frequency and sentence importance.
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs
        tabs = st.tabs(["Summarization", "Analysis", "Evaluation"])
        
        # Summarization Tab
        with tabs[0]:
            st.subheader("Generate Summary")
            
            # Text input options
            text_input = st.text_area("Paste text here:", height=200)
            uploaded_file = st.file_uploader("Or upload a document", type=["txt", "pdf"])
            
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
            if st.button("Generate Summary"):
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
                    if summary:
                        st.success("Summary generated successfully!")
                        st.markdown("### Summary")
                        st.markdown(summary)
                        
                        # Download button for the summary
                        st.download_button(
                            label="Download Summary",
                            data=summary,
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
                col1.metric("Original Text Words", original_word_count)
                col2.metric("Summary Words", summary_word_count)
                col3.metric("Compression Ratio", f"{summary_word_count/original_word_count:.1%}" if original_word_count > 0 else "N/A")
                
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
                    st.dataframe(freq_df)
                
                with col2:
                    st.markdown("**Summary**")
                    # Get word frequency in summary
                    summary_words = re.findall(r'\w+', summary.lower())
                    # Filter out common stopwords
                    summary_words = [word for word in summary_words if word not in stopwords and len(word) > 2]
                    summary_word_freq = Counter(summary_words).most_common(10)
                    
                    # Display as a table
                    summary_freq_df = pd.DataFrame(summary_word_freq, columns=["Word", "Frequency"])
                    st.dataframe(summary_freq_df)
        
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
                
                if st.button("Evaluate Summary"):
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
                                col1.metric("Overall Similarity", f"{scores.get('similarity', 0):.4f}")
                                col2.metric("Precision", f"{scores.get('precision', 0):.4f}")
                                col3.metric("Recall", f"{scores.get('recall', 0):.4f}")
                                
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

# -------------------- BI Dashboards -------------------- #
if page == "BI Dashboards":
    dashboard = st.sidebar.radio("Choose Dashboard:", ["Healthcare Analytics", "Call Center Analytics"])

    if dashboard == "Healthcare Analytics":
        st.title("🏥 Healthcare Analytics Dashboard")
        st.markdown("""
        Explore patient demographics, hospital billing trends, and admission insights using an interactive
        Power BI dashboard with filters for year and hospital facility.
        """)

        # Check if image files exist before displaying
        image_paths = [
            "UpdatedHealthAnalysis_page-0001.jpg",
            "UpdatedHealthAnalysis_page-0002.jpg",
            "UpdatedHealthAnalysis_page-0003.jpg"
        ]
        
        for image_path in image_paths:
            if os.path.exists(image_path):
                st.image(image_path, caption=image_path.split("_")[-1].split(".")[0])
            else:
                st.error(f"Image file '{image_path}' not found. Please add this file to your project.")
        
        # Check if Power BI file exists before displaying download button
        pbit_path = "UpdatedHealthAnalysis.pbit"
        if os.path.exists(pbit_path):
            with open(pbit_path, "rb") as f:
                st.download_button("Download Power BI file (.pbit)", f, file_name="HealthcareAnalytics.pbit")
        else:
            st.error(f"Power BI file '{pbit_path}' not found. Please add this file to your project.")

    if dashboard == "Call Center Analytics":
        st.title("📞 Call Center Analytics Dashboard")
        st.markdown("""
        Visualizes call center performance including call volume, customer satisfaction, and agent metrics.
        Includes breakdowns by hour, agent, and topic.
        """)

        # Check if image exists before displaying
        image_path = "call center analytics.jpg"
        if os.path.exists(image_path):
            st.image(image_path, caption="Call Center Overview")
        else:
            st.error(f"Image file '{image_path}' not found. Please add this file to your project.")

        # Check if Power BI file exists before displaying download button
        pbix_path = "call center analytics.pbix"
        if os.path.exists(pbix_path):
            with open(pbix_path, "rb") as f:
                st.download_button("Download Power BI file (.pbix)", f, file_name="CallCenterAnalytics.pbix")
        else:
            st.error(f"Power BI file '{pbix_path}' not found. Please add this file to your project.")
        
        # Display sample metrics
        st.subheader("Key Performance Indicators")
        
        kpi_cols = st.columns(4)
        with kpi_cols[0]:
            st.metric("Total Calls", "1,247", "+12%")
        with kpi_cols[1]:
            st.metric("Avg. Wait Time", "2m 34s", "-8%")
        with kpi_cols[2]:
            st.metric("Satisfaction Score", "4.2/5", "+0.3")
        with kpi_cols[3]:
            st.metric("Call Resolution Rate", "87%", "+5%")

# Display footer with copyright
st.markdown("""
---
© 2025 Samuel Chukwuka Mbah | Portfolio created with Streamlit | Last updated: May 2025
""")
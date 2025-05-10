
# -------------------- Global Imports -------------------- #
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
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
from sklearn.metrics import accuracy_score, confusion_matrix
from wordcloud import WordCloud
from PyPDF2 import PdfReader
import docx2txt
from PIL import Image
import requests
import time
import re
from collections import Counter
import io

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

def generate_wordcloud(data):
    wc = WordCloud(background_color='white', max_words=100, width=800, height=400).generate(" ".join(data))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# -------------------- Config -------------------- #
st.set_page_config(page_title="🧠 ML & BI Portfolio", page_icon="🧠", layout="wide")
st.sidebar.title("ML & BI Portfolio")
page = st.sidebar.radio("Select Page:", ["About Me", "Machine Learning Projects", "BI Dashboards"])

# -------------------- About Me -------------------- #
# [unchanged code omitted for brevity]
# -------------------- About Me -------------------- #
if page == "About Me":
    st.title("👤 About Me")
    # Load and crop the image
    original_img = Image.open("IMG_4202.jpg")
    width, height = original_img.size

    # Crop from the top-left corner to half height (adjust as needed)
    cropped_img = original_img.crop((0, 0, width, int(height * 0.6)))
    # Display cropped image
    st.image(cropped_img, width=200)
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
        st.title("🎼️ IMDB Sentiment Analysis")
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

        df = pd.read_csv("train.csv")
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
            st.subheader("Class Distribution")
            st.bar_chart(df["label"].value_counts())
            st.subheader("Word Cloud")
            generate_wordcloud(df["text"])

        with tabs[2]:
            st.write(f"Accuracy: {accuracy_score(y_val, y_pred):.2f}")

    # Additional projects remain unchanged...



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
    # ------------------- Text Summarization ------------------- #

    
        
    if project == "Text Summarization":
        

        # Try to import PyPDF2 for PDF support
        try:
            from PyPDF2 import PdfReader
            PDF_SUPPORT = True
        except ImportError:
            PDF_SUPPORT = False
            st.warning("PyPDF2 is not installed. PDF support is disabled.")

        # Simple version that doesn't rely on external dependencies
        # We'll implement our own text summarization logic

        # Set page configuration
        st.set_page_config(
            page_title="Text Summarization App", 
            page_icon="📝",
            layout="wide"
        )

        # Extract text from uploaded file
        def extract_text_from_file(uploaded_file):
            try:
                if uploaded_file.type == "text/plain":
                    return uploaded_file.read().decode("utf-8")
                elif uploaded_file.type == "application/pdf" and PDF_SUPPORT:
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

        # Custom text summarization function using sentence ranking
        @st.cache_data
        def generate_summary(text, max_length=150, min_length=40):
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

        # Calculate simple text similarity score
        def calculate_similarity_score(summary, reference):
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

        # Main app function
        def main():
            # App header
            st.title("📝 Text Summarization")
            st.markdown("""
            This tool summarizes text using an extractive approach that identifies and extracts key sentences.
            Simply paste your text or upload a .txt file to generate a concise summary.
            """)
            
            # Initialize session state variables if they don't exist
            if 'generated_summary' not in st.session_state:
                st.session_state.generated_summary = ""
            
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
                        if st.session_state.generated_summary:
                            st.success("Summary generated successfully!")
                            st.markdown("### Summary")
                            st.markdown(st.session_state.generated_summary)
                            
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

        if __name__ == "__main__":
            try:
                main()
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.markdown("Try reloading the page or providing different input text.")


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

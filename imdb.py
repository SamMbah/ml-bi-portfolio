import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from wordcloud import WordCloud

# ---------------------- Sidebar Project Selector ---------------------- #
st.sidebar.title("ML Projects Portfolio")
project = st.sidebar.radio("Choose a project:", ["IMDB Sentiment Analysis", "Malware URL Detection", "Loan Default Prediction"])


# ---------------------- IMDB Sentiment Analysis ---------------------- #
if project == "IMDB Sentiment Analysis":
    st.title("🎬 IMDB Sentiment Analysis")
    st.write("Sentiment analysis on IMDB movie reviews using NLP techniques.")

    # Load model and vectorizer
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, "sentiment_model.pkl")
    vectorizer_path = os.path.join(current_dir, "tfidf_vectorizer.pkl")
    train_data_path = os.path.join(current_dir, "train.csv")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    df = pd.read_csv(train_data_path)

    # Split for evaluation
    X_train, X_val, y_train, y_val = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    X_val_vec = vectorizer.transform(X_val)
    y_pred = model.predict(X_val_vec)

    tabs_imdb = st.tabs(["Prediction", "EDA", "Model Performance"])

    # Prediction
    with tabs_imdb[0]:
        st.subheader("Make a Prediction")
        user_input = st.text_area("Enter a movie review:")
        if st.button("Predict Sentiment"):
            if user_input.strip() == "":
                st.warning("Please enter a review text.")
            else:
                input_vec = vectorizer.transform([user_input])
                prediction = model.predict(input_vec)[0]
                probability = model.predict_proba(input_vec)[0][prediction]
                sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
                st.header(f"Prediction: {sentiment}")
                st.write(f"Confidence: {probability:.2f}")

    # EDA
    with tabs_imdb[1]:
        st.subheader("Exploratory Data Analysis")
        st.write("### Sentiment Distribution")
        sentiment_counts = df['label'].value_counts().rename({0: "Negative", 1: "Positive"})
        fig, ax = plt.subplots()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax)
        st.pyplot(fig)

        st.write("### Text Length Distribution")
        df['text_length'] = df['text'].apply(len)
        fig2, ax2 = plt.subplots()
        sns.histplot(df['text_length'], bins=50, kde=True, ax=ax2)
        st.pyplot(fig2)

        st.write("### Word Clouds")
        positive_text = " ".join(df[df['label'] == 1]['text'])
        negative_text = " ".join(df[df['label'] == 0]['text'])
        wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
        wordcloud_negative = WordCloud(width=800, height=400, background_color='black').generate(negative_text)

        st.write("**Positive Reviews**")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.imshow(wordcloud_positive, interpolation='bilinear')
        ax3.axis('off')
        st.pyplot(fig3)

        st.write("**Negative Reviews**")
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        ax4.imshow(wordcloud_negative, interpolation='bilinear')
        ax4.axis('off')
        st.pyplot(fig4)

    # Model Performance
    with tabs_imdb[2]:
        st.subheader("Model Performance")
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        st.write("### Accuracy Metrics")
        st.write(f"**Accuracy:** {accuracy:.2f}")
        st.write(f"**Precision:** {precision:.2f}")
        st.write(f"**Recall:** {recall:.2f}")
        st.write(f"**F1 Score:** {f1:.2f}")

        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_val, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
        st.pyplot(fig_cm)


# ---------------------- Malware URL Detection ---------------------- #
if project == "Malware URL Detection":
    st.title("🛡️ Malware URL Detection")
    st.write("Classify URLs as Safe or Malicious using ML model.")

    df = pd.read_csv("malicious_phish.csv")
    df['label'] = df['type'].apply(lambda x: "Safe" if x == "benign" else "Malicious")

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

    tabs_url = st.tabs(["Prediction", "EDA", "Model Performance"])

    with tabs_url[0]:
        st.subheader("Make a Prediction")
        url_input = st.text_input("Enter a URL:")
        if st.button("Predict URL"):
            if url_input.strip() == "":
                st.warning("Please enter a URL.")
            else:
                input_vec = vectorizer.transform([url_input])
                prediction = model.predict(input_vec)[0]
                result = "Safe ✅" if prediction == 0 else "Malicious 🚨"
                st.header(f"Prediction: {result}")

    with tabs_url[1]:
        st.subheader("Exploratory Data Analysis")
        st.write("### URL Type Distribution")
        counts = df["label"].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=counts.index, y=counts.values, ax=ax)
        st.pyplot(fig)

        st.write("### URL Length Distribution")
        df["url_length"] = df["url"].apply(len)
        fig2, ax2 = plt.subplots()
        sns.histplot(df["url_length"], bins=50, kde=True, ax=ax2)
        st.pyplot(fig2)

        st.write("### Word Clouds")
        malicious_text = " ".join(df[df["label"] == "Malicious"]["url"])
        safe_text = " ".join(df[df["label"] == "Safe"]["url"])
        wordcloud_mal = WordCloud(width=800, height=400, background_color='black').generate(malicious_text)
        wordcloud_safe = WordCloud(width=800, height=400, background_color='white').generate(safe_text)

        st.write("**Malicious URLs**")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.imshow(wordcloud_mal, interpolation='bilinear')
        ax3.axis('off')
        st.pyplot(fig3)

        st.write("**Safe URLs**")
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        ax4.imshow(wordcloud_safe, interpolation='bilinear')
        ax4.axis('off')
        st.pyplot(fig4)

    with tabs_url[2]:
        st.subheader("Model Performance")
        X_test_vec = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_vec)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        st.write("### Accuracy Metrics")
        st.write(f"**Accuracy:** {accuracy:.2f}")
        st.write(f"**Precision:** {precision:.2f}")
        st.write(f"**Recall:** {recall:.2f}")
        st.write(f"**F1 Score:** {f1:.2f}")

        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Safe", "Malicious"], yticklabels=["Safe", "Malicious"])
        st.pyplot(fig_cm)


# ---------------------- Loan Default Prediction ---------------------- #
if project == "Loan Default Prediction":
    st.title("💳 Loan Default Prediction")
    st.write("Predict whether a customer will fully pay or default on their loan.")

    df = pd.read_csv("clean_loan_data.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    tabs_loan = st.tabs(["Prediction", "EDA", "Model Performance"])

    with tabs_loan[0]:
        st.subheader("Make a Prediction")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        annual_inc = st.number_input("Annual Income", min_value=0, max_value=500000, value=50000)
        loan_amnt = st.number_input("Loan Amount", min_value=0, max_value=100000, value=5000)
        emp_length = st.number_input("Employment Length", min_value=0, max_value=50, value=5)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)

        input_data = np.array([[age, annual_inc, loan_amnt, emp_length, credit_score]])
        prediction = model.predict(input_data)[0]
        result = "Charged Off 🚨 (Default)" if prediction == 1 else "Fully Paid ✅"
        st.header(f"Prediction: {result}")

    with tabs_loan[1]:
        st.subheader("Exploratory Data Analysis")
        st.write("### Loan Status Distribution")
        counts = df["target"].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=counts.index, y=counts.values, ax=ax)
        st.pyplot(fig)

        st.write("### Loan Amount Distribution")
        fig2, ax2 = plt.subplots()
        sns.histplot(df["loan_amnt"], bins=20, kde=True, ax=ax2)
        st.pyplot(fig2)

        st.write("### Correlation Heatmap")
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax3)
        st.pyplot(fig3)

    with tabs_loan[2]:
        st.subheader("Model Performance")
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        st.write("### Accuracy Metrics")
        st.write(f"**Accuracy:** {accuracy:.2f}")
        st.write(f"**Precision:** {precision:.2f}")
        st.write(f"**Recall:** {recall:.2f}")
        st.write(f"**F1 Score:** {f1:.2f}")

        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fully Paid", "Charged Off"], yticklabels=["Fully Paid", "Charged Off"])
        st.pyplot(fig_cm)

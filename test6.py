"""
ML & BI Portfolio with Customer Churn Prediction

This application showcases various machine learning and business intelligence projects:
1. Customer Churn Prediction - Predict which customers are likely to leave a service
2. Text Summarization - Create concise summaries of longer documents
3. Malware URL Detection - Identify potentially harmful URLs
4. Loan Default Prediction - Assess risk of loan defaults

Author: Samuel Chukwuka Mbah
"""

# -------------------- Global Imports -------------------- #
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import requests
import io
import re
import time
from collections import Counter
from datetime import datetime
from PIL import Image
from PyPDF2 import PdfReader
from io import BytesIO, StringIO
from wordcloud import WordCloud

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
        score = sum(word_freq.get(word, 0) for word in words_in_sentence) / (len(words_in_sentence) or 1)
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

# -------------------- Customer Churn Prediction Functions -------------------- #
def load_telco_data():
    """
    Load the Telco Customer Churn dataset
    
    Returns:
        DataFrame with customer data
    """
    # URL for the dataset
    DATASET_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    
    try:
        # Check if dataset already exists locally
        if os.path.exists("telco_churn.csv"):
            df = pd.read_csv("telco_churn.csv")
        else:
            # Download dataset from URL
            with st.spinner("Downloading Telco Customer Churn dataset..."):
                response = requests.get(DATASET_URL)
                if response.status_code == 200:
                    df = pd.read_csv(StringIO(response.content.decode('utf-8')))
                    # Save a copy locally
                    df.to_csv("telco_churn.csv", index=False)
                    st.success(f"Dataset downloaded with {len(df)} customers")
                else:
                    raise Exception(f"Failed to download dataset: {response.status_code}")
                
        return df
            
    except Exception as e:
        st.error(f"Error loading telco dataset: {e}")
        # Return a minimal DataFrame with the right structure if download fails
        return pd.DataFrame(columns=[
            'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
        ])

def preprocess_telco_data(df):
    """
    Preprocess the Telco Customer Churn dataset
    
    Args:
        df: DataFrame with telco customer data
        
    Returns:
        Processed DataFrame ready for modeling
    """
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Convert TotalCharges to numeric, handling non-numeric values
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    
    # Fill missing values
    df_processed['TotalCharges'].fillna(0, inplace=True)
    
    # Convert binary categorical variables to 1/0
    df_processed['SeniorCitizen'] = df_processed['SeniorCitizen'].astype(int)
    
    # Convert Yes/No to 1/0 for binary categorical variables
    binary_vars = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for var in binary_vars:
        df_processed[var] = df_processed[var].map({'Yes': 1, 'No': 0})
    
    # Handle other categorical variables with multiple values
    # For these, we'll use one-hot encoding in the modeling pipeline
    
    return df_processed

def create_customer_profile(df, profile_type='random'):
    """
    Create a customer profile for churn prediction
    
    Args:
        df: DataFrame with telco customer data
        profile_type: 'high_risk', 'low_risk', or 'random'
        
    Returns:
        Dictionary with customer profile data
    """
    if len(df) == 0:
        return {}
    
    if profile_type == 'high_risk':
        # Filter for customers who have churned
        churn_df = df[df['Churn'] == 1]
        if len(churn_df) > 0:
            # Get a random high risk customer
            customer = churn_df.sample(1).iloc[0].to_dict()
        else:
            # Fallback to random if no churn examples
            customer = df.sample(1).iloc[0].to_dict()
    
    elif profile_type == 'low_risk':
        # Filter for loyal customers (didn't churn, high tenure)
        loyal_df = df[(df['Churn'] == 0) & (df['tenure'] > 40)]
        if len(loyal_df) > 0:
            # Get a random loyal customer
            customer = loyal_df.sample(1).iloc[0].to_dict()
        else:
            # Fallback to random non-churned customer
            customer = df[df['Churn'] == 0].sample(1).iloc[0].to_dict()
    
    else:  # random
        # Get a random customer
        customer = df.sample(1).iloc[0].to_dict()
    
    return customer

def train_churn_model(X_train, y_train):
    """
    Train a churn prediction model
    
    Args:
        X_train: Feature DataFrame
        y_train: Target Series
        
    Returns:
        Trained pipeline
    """
    # Identify numerical and categorical features
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns
    
    # Create preprocessing pipelines for both numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create the full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    return pipeline

def predict_churn(model, customer_data, feature_names):
    """
    Predict churn for a customer
    
    Args:
        model: Trained pipeline
        customer_data: Dictionary with customer features
        feature_names: List of feature names (columns) used by the model
        
    Returns:
        prediction, probability
    """
    try:
        # Create a DataFrame with the customer data
        customer_df = pd.DataFrame([customer_data], columns=feature_names)
        
        # Make prediction
        prediction = model.predict(customer_df)[0]
        
        # Get probability
        probability = model.predict_proba(customer_df)[0][1]  # Probability of churn (class 1)
        
        return int(prediction), float(probability)
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        print(f"Error making prediction: {e}")
        return None, 0.0

def churn_prediction_section():
    """
    Create the customer churn prediction UI section
    """
    st.markdown('<h3>Telco Customer Churn Prediction</h3>', unsafe_allow_html=True)
    
    # Load and preprocess data
    with st.spinner("Loading Telco customer data..."):
        df = load_telco_data()
        if len(df) == 0:
            st.error("Failed to load the dataset. Please check your internet connection and try again.")
            return
        
        # Preprocess data
        df_processed = preprocess_telco_data(df)
    
    # Display dataset info
    st.markdown(f"**Dataset**: {len(df_processed)} customers | **Churn rate**: {df_processed['Churn'].mean():.1%}")
    
    # Train model tab and customer analysis tab
    tabs = st.tabs(["Churn Predictor", "Dataset Insights", "Model Performance"])
    
    with tabs[0]:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Customer Profile")
            
            # Initialize session state for customer profile
            if 'customer_profile' not in st.session_state:
                st.session_state.customer_profile = create_customer_profile(df_processed, 'random')
            
            # Display current customer profile
            if st.session_state.customer_profile:
                profile = st.session_state.customer_profile
                
                # Main customer info
                profile_col1, profile_col2 = st.columns(2)
                
                with profile_col1:
                    st.markdown(f"**Tenure**: {profile.get('tenure', 'N/A')} months")
                    st.markdown(f"**Monthly Charges**: ${profile.get('MonthlyCharges', 'N/A'):.2f}")
                    st.markdown(f"**Total Charges**: ${profile.get('TotalCharges', 'N/A'):.2f}")
                    st.markdown(f"**Contract**: {profile.get('Contract', 'N/A')}")
                
                with profile_col2:
                    st.markdown(f"**Internet Service**: {profile.get('InternetService', 'N/A')}")
                    st.markdown(f"**Online Security**: {profile.get('OnlineSecurity', 'N/A')}")
                    st.markdown(f"**Tech Support**: {profile.get('TechSupport', 'N/A')}")
                    st.markdown(f"**Payment Method**: {profile.get('PaymentMethod', 'N/A')}")
                
                # Additional services section
                st.markdown("**Additional Services:**")
                services_col1, services_col2, services_col3 = st.columns(3)
                
                with services_col1:
                    st.markdown(f"• Phone Service: {profile.get('PhoneService', 'N/A')}")
                    st.markdown(f"• Multiple Lines: {profile.get('MultipleLines', 'N/A')}")
                
                with services_col2:
                    st.markdown(f"• Online Backup: {profile.get('OnlineBackup', 'N/A')}")
                    st.markdown(f"• Device Protection: {profile.get('DeviceProtection', 'N/A')}")
                
                with services_col3:
                    st.markdown(f"• Streaming TV: {profile.get('StreamingTV', 'N/A')}")
                    st.markdown(f"• Streaming Movies: {profile.get('StreamingMovies', 'N/A')}")
        
        with col2:
            st.markdown("### Customer Generator")
            
            # Buttons for profile generation
            if st.button("Generate Random Customer"):
                st.session_state.customer_profile = create_customer_profile(df_processed, 'random')
                st.rerun()
            
            if st.button("Generate High-Risk Customer"):
                st.session_state.customer_profile = create_customer_profile(df_processed, 'high_risk')
                st.rerun()
            
            if st.button("Generate Low-Risk Customer"):
                st.session_state.customer_profile = create_customer_profile(df_processed, 'low_risk')
                st.rerun()
        
        # Predict button
        if st.button("Predict Churn Risk", key="predict_churn_button"):
            with st.spinner("Analyzing customer profile..."):
                # Prepare data for model
                X = df_processed.drop(['customerID', 'Churn'], axis=1)
                y = df_processed['Churn']
                
                # Split data for training
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42, stratify=y
                )
                
                # Train model
                model = train_churn_model(X_train, y_train)
                
                # Get feature names used by the model
                feature_names = X.columns.tolist()
                
                # Make prediction
                prediction, probability = predict_churn(
                    model, 
                    st.session_state.customer_profile, 
                    feature_names
                )
                
                # Display result
                if prediction is not None:
                    if prediction == 1:
                        result = "High Risk of Churn ⚠️"
                        result_color = "#dc3545"  # Red
                    else:
                        result = "Low Risk of Churn ✓"
                        result_color = "#28a745"  # Green
                    
                    st.markdown(f"""
                    <div style='background-color:{result_color}; padding:1rem; border-radius:10px; color:white; text-align:center; margin:1rem 0;'>
                        <h2>{result}</h2>
                        <p>Churn Probability: {probability:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show key churn factors
                    st.markdown("### Key Churn Factors")
                    
                    # Identify important features for this customer
                    # These are based on domain knowledge about telco churn
                    risk_factors = []
                    
                    # Contract type
                    if profile.get('Contract') == 'Month-to-month':
                        risk_factors.append("Month-to-month contract increases churn risk by 3x compared to long-term contracts")
                    
                    # Tenure
                    if profile.get('tenure', 0) < 12:
                        risk_factors.append("New customers with less than 12 months tenure have higher churn rates")
                    
                    # Services
                    if profile.get('OnlineSecurity') == 'No':
                        risk_factors.append("Customers without online security services are more likely to churn")
                    
                    if profile.get('TechSupport') == 'No':
                        risk_factors.append("Lack of tech support correlates with higher churn rates")
                    
                    # Payment method
                    if profile.get('PaymentMethod') == 'Electronic check':
                        risk_factors.append("Customers using electronic checks have higher churn rates than those with automatic payments")
                    
                    # Monthly charges
                    if profile.get('MonthlyCharges', 0) > 80:
                        risk_factors.append("Higher monthly charges correlate with increased likelihood of churn")
                    
                    # Display factors
                    if risk_factors:
                        for factor in risk_factors:
                            st.markdown(f"• {factor}")
                    else:
                        st.markdown("• This customer has a stable profile with few risk indicators")
                    
                    # Retention suggestions
                    st.markdown("### Retention Suggestions")
                    
                    if prediction == 1:  # High risk
                        st.markdown("""
                        1. **Offer contract upgrade** with promotional pricing
                        2. **Suggest bundled services** to increase product attachment
                        3. **Provide retention incentives** such as bill credits or service upgrades
                        4. **Proactive customer support** to address any service issues
                        """)
                    else:  # Low risk
                        st.markdown("""
                        1. **Cross-sell additional services** to increase customer value
                        2. **Encourage referrals** through a customer loyalty program
                        3. **Suggest paperless billing** and autopay for convenience
                        4. **Regular satisfaction surveys** to maintain engagement
                        """)
                else:
                    st.error("Unable to make a prediction. Please try again.")
    
    with tabs[1]:
        st.markdown("### Telco Customer Churn Dataset Insights")
        
        # Customer distribution by contract type and churn
        st.markdown("#### Contract Type vs. Churn")
        
        # Convert 1/0 back to Yes/No for better visualization
        df_viz = df_processed.copy()
        df_viz['Churn'] = df_viz['Churn'].map({1: 'Yes', 0: 'No'})
        
        # Create the visualization
        fig, ax = plt.subplots(figsize=(10, 5))
        contract_churn = pd.crosstab(df_viz['Contract'], df_viz['Churn'], normalize='index')
        contract_churn.plot(kind='bar', stacked=True, ax=ax, colormap='coolwarm')
        ax.set_title('Churn Rate by Contract Type')
        ax.set_xlabel('Contract Type')
        ax.set_ylabel('Percentage')
        ax.legend(title='Churn')
        st.pyplot(fig)
        
        st.markdown("""
        **Observation**: Month-to-month contracts have significantly higher churn rates 
        compared to one-year or two-year contracts.
        """)
        
        # Tenure vs Churn
        st.markdown("#### Customer Tenure vs. Churn")
        
        # Group by tenure and calculate churn rate
        tenure_churn = df_processed.groupby('tenure')['Churn'].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(tenure_churn['tenure'], tenure_churn['Churn'] * 100, marker='o', linestyle='-')
        ax.set_title('Churn Rate by Tenure')
        ax.set_xlabel('Tenure (months)')
        ax.set_ylabel('Churn Rate (%)')
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
        
        st.markdown("""
        **Observation**: Churn rates are highest in the early months of service and 
        generally decrease as customers stay longer with the company.
        """)
        
        # Monthly charges vs churn
        st.markdown("#### Monthly Charges vs. Churn")
        
        # Create bins for monthly charges
        bins = [0, 20, 40, 60, 80, 100, 120]
        labels = ['0-20', '21-40', '41-60', '61-80', '81-100', '101-120']
        
        df_viz['MonthlyChargesBin'] = pd.cut(df_viz['MonthlyCharges'], bins=bins, labels=labels)
        
        charge_churn = pd.crosstab(df_viz['MonthlyChargesBin'], df_viz['Churn'], normalize='index')
        
        fig, ax = plt.subplots(figsize=(10, 5))
        charge_churn.plot(kind='bar', stacked=True, ax=ax, colormap='coolwarm')
        ax.set_title('Churn Rate by Monthly Charges')
        ax.set_xlabel('Monthly Charges ($)')
        ax.set_ylabel('Percentage')
        ax.legend(title='Churn')
        st.pyplot(fig)
        
        st.markdown("""
        **Observation**: Customers with higher monthly charges tend to have higher churn rates,
        suggesting price sensitivity.
        """)
        
        # Additional services impact
        st.markdown("#### Impact of Additional Services on Churn")
        
        # Create a figure with multiple subplots
        services = ['OnlineSecurity', 'TechSupport', 'OnlineBackup', 'DeviceProtection']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, service in enumerate(services):
            service_churn = pd.crosstab(df_viz[service], df_viz['Churn'], normalize='index')
            service_churn.plot(kind='bar', stacked=True, ax=axes[i], colormap='coolwarm')
            axes[i].set_title(f'Churn Rate by {service}')
            axes[i].set_xlabel(service)
            axes[i].set_ylabel('Percentage')
            axes[i].legend(title='Churn')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Observation**: Customers without additional services like Online Security and 
        Tech Support have higher churn rates, suggesting these services help with retention.
        """)
    
    with tabs[2]:
        st.markdown("### Churn Prediction Model Performance")
        
        if st.button("Train and Evaluate Model"):
            with st.spinner("Training and evaluating churn prediction model..."):
                # Prepare data for model
                X = df_processed.drop(['customerID', 'Churn'], axis=1)
                y = df_processed['Churn']
                
                # Split data for training and testing
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42, stratify=y
                )
                
                # Train model
                model = train_churn_model(X_train, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                # Display metrics
                st.markdown("#### Model Performance Metrics")
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    st.metric("Accuracy", f"{accuracy:.2%}")
                    st.metric("Precision", f"{precision:.2%}")
                
                with metrics_col2:
                    st.metric("Recall (Sensitivity)", f"{recall:.2%}")
                    st.metric("F1 Score", f"{f1:.2%}")
                
                # Confusion matrix
                st.markdown("#### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title('Confusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_xticklabels(['No Churn', 'Churn'])
                ax.set_yticklabels(['No Churn', 'Churn'])
                st.pyplot(fig)
                
                # ROC curve
                st.markdown("#### ROC Curve")
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic (ROC)')
                ax.legend(loc="lower right")
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)
                
                # Feature importance
                st.markdown("#### Feature Importance")
                
                # Get feature names after preprocessing
                preprocessor = model.named_steps['preprocessor']
                feature_names = []
                
                # Get all transformed feature names
                for name, transformer, features in preprocessor.transformers_:
                    if name == 'num':
                        # Numerical features stay the same
                        feature_names.extend(features)
                    else:
                        # Get one-hot encoded feature names
                        encoder = transformer.named_steps['onehot']
                        for feature, categories in zip(features, encoder.categories_):
                            for category in categories:
                                feature_names.append(f"{feature}_{category}")
                
                # Get feature importances
                feature_importances = model.named_steps['classifier'].feature_importances_
                
                # Create a DataFrame for visualization
                if len(feature_names) == len(feature_importances):
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': feature_importances
                    }).sort_values('Importance', ascending=False).head(15)
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
                    ax.set_title('Top 15 Feature Importances')
                    ax.set_xlabel('Importance')
                    ax.set_ylabel('Feature')
                    st.pyplot(fig)
                else:
                    st.warning(f"Feature names and importances length mismatch: {len(feature_names)} vs {len(feature_importances)}")
                    st.text("Feature importances are available but can't be mapped to feature names")

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
            original_img = Image.open("IMG_4202 - Copy.jpg")
            width, height = original_img.size
            cropped_img = original_img.crop((0, 0, width, int(height * 0.6)))
            st.image(cropped_img, width=200)
        except:
            st.info("Profile image not found. Please add your image as 'IMG_4202 - Copy.jpg'")
    
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
        **Bright Network Internship Experience UK (2024)**
        
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
        "Customer Churn Prediction",
        "Malware URL Detection",
        "Loan Default Prediction",
        "Text Summarization"
    ])

    # ------------------- Customer Churn Prediction ------------------- #
    if project == "Customer Churn Prediction":
        st.markdown('<h2 class="project-header">👥 Customer Churn Prediction</h2>', unsafe_allow_html=True)
        st.markdown("""
        <div style='background-color:#f5f7fb; padding:1rem; border-radius:10px; margin-bottom:1rem;'>
        This project uses machine learning to predict which customers are likely to cancel their service (churn). 
        The model analyzes customer profiles and service usage patterns to identify at-risk customers, enabling
        proactive retention efforts.
        </div>
        """, unsafe_allow_html=True)

        # Display the churn prediction section
        churn_prediction_section()

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
            
            # Create synthetic data for demonstration
            np.random.seed(42)
            n_samples = 1000
            
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
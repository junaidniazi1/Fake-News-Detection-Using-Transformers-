import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import pandas as pd
import os
from io import BytesIO
import base64

# Set Streamlit page config
st.set_page_config(
    page_title="📰 Fake News Detector", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
def load_css():
    st.markdown("""
    <style>
    .stApp {
        background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), 
                         url('https://images.unsplash.com/photo-1504711434969-e33886168f5c?ixlib=rb-4.0.3&auto=format&fit=crop&w=2000&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
    }
    
    .main-header h1 {
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .main-header p {
        font-size: 1.2rem;
        color: #e0e0e0;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 0.5rem;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.3);
        color: #1f1f1f;
    }
    
    .prediction-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin: 1rem 0;
    }
    
    .fake-result {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.2), rgba(255, 99, 71, 0.2));
        border-left: 5px solid #ff6b6b;
    }
    
    .real-result {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.2), rgba(46, 204, 113, 0.2));
        border-left: 5px solid #4caf50;
    }
    
    .upload-section {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        border: 2px dashed rgba(255, 255, 255, 0.3);
        text-align: center;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stTextArea textarea {
        background-color: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        color: white;
    }
    
    .stTextArea textarea::placeholder {
        color: rgba(255, 255, 255, 0.7);
    }
    
    .stButton button {
        background: linear-gradient(45deg, #4ecdc4, #45b7d1);
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        color: white;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: rgba(255, 255, 255, 0.7);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# Load model and tokenizer
@st.cache_resource
def load_model():
    try:
        model_path = r"C:\Users\kashif-pc\Desktop\DATA ANYALSYIS PROJECT\ML projects\FAKE_NEWS\fake_news_bert"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return tokenizer, model, True
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, False

tokenizer, model, model_loaded = load_model()

# Prediction function
def predict_single(text):
    if not model_loaded:
        return None, None
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().numpy()
    predicted_class = np.argmax(probs)
    return predicted_class, probs

def predict_batch(texts):
    if not model_loaded:
        return None
    
    predictions = []
    probabilities = []
    
    for text in texts:
        if pd.isna(text) or text.strip() == "":
            predictions.append(None)
            probabilities.append([None, None])
        else:
            pred, prob = predict_single(str(text))
            predictions.append(pred)
            probabilities.append(prob)
    
    return predictions, probabilities

# Header
st.markdown("""
<div class="main-header">
    <h1>📰 AI Fake News Detector</h1>
    <p>🤖 Powered by BERT | Real-time Detection & Batch Processing</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Model could not be loaded. Please check the model path.")
    st.stop()

# Create tabs
tab1, tab2 = st.tabs(["📝 Single Text Analysis", "📊 Batch CSV Processing"])

with tab1:
    st.markdown("### ✍️ Enter News Text for Analysis")
    
    user_input = st.text_area(
        "",
        height=150,
        placeholder="Paste your news headline or article text here..."
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        detect_btn = st.button("🔍 Analyze News", use_container_width=True)
    
    if detect_btn:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            with st.spinner("🧠 Analyzing..."):
                pred, prob = predict_single(user_input)
                
                if pred is not None:
                    label = "FAKE" if pred == 1 else "REAL"
                    confidence = prob[pred] * 100
                    
                    # Results card
                    card_class = "fake-result" if pred == 1 else "real-result"
                    icon = "🚨" if pred == 1 else "✅"
                    
                    st.markdown(f"""
                    <div class="prediction-card {card_class}">
                        <h2 style="color: white; text-align: center; margin-bottom: 1rem;">
                            {icon} Prediction: {label}
                        </h2>
                        <p style="color: white; text-align: center; font-size: 1.2rem;">
                            Confidence: {confidence:.2f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Progress bar
                    st.progress(int(confidence), text=f"{label} - {confidence:.2f}% confidence")
                    
                    # Detailed probabilities
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: #4caf50;">✅ REAL</h4>
                            <h3 style="color: white;">{prob[0]*100:.2f}%</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: #ff6b6b;">🚨 FAKE</h4>
                            <h3 style="color: white;">{prob[1]*100:.2f}%</h3>
                        </div>
                        """, unsafe_allow_html=True)

with tab2:
    st.markdown("### 📊 Batch Processing with CSV")
    
    st.markdown("""
    <div class="upload-section">
        <h4 style="color: white; margin-bottom: 1rem;">📁 Upload Your CSV File</h4>
        <p style="color: rgba(255,255,255,0.8); margin-bottom: 1rem;">
            Your CSV should contain a column with news text (e.g., 'text', 'news', 'headline', 'content')
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="Upload a CSV file containing news text for batch classification"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
            
            # Show preview
            st.markdown("#### 👀 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Column selection
            text_columns = [col for col in df.columns if df[col].dtype == 'object']
            
            if text_columns:
                selected_column = st.selectbox(
                    "📋 Select the column containing news text:",
                    text_columns,
                    index=0
                )
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    process_btn = st.button("🚀 Process All News", use_container_width=True)
                
                if process_btn:
                    with st.spinner("🔄 Processing all news articles..."):
                        # Get predictions
                        texts = df[selected_column].tolist()
                        predictions, probabilities = predict_batch(texts)
                        
                        # Add results to dataframe
                        df['prediction'] = ['FAKE' if p == 1 else 'REAL' if p == 0 else 'ERROR' for p in predictions]
                        df['confidence'] = [prob[p]*100 if p is not None and prob[p] is not None else 0 for p, prob in zip(predictions, probabilities)]
                        df['real_probability'] = [prob[0]*100 if prob[0] is not None else 0 for prob in probabilities]
                        df['fake_probability'] = [prob[1]*100 if prob[1] is not None else 0 for prob in probabilities]
                        
                        # Display results summary
                        st.markdown("### 📈 Results Summary")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        total_articles = len(df)
                        fake_count = sum(1 for p in predictions if p == 1)
                        real_count = sum(1 for p in predictions if p == 0)
                        error_count = sum(1 for p in predictions if p is None)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="color: white;">📰 Total</h4>
                                <h3 style="color: #45b7d1;">{total_articles}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="color: white;">✅ Real</h4>
                                <h3 style="color: #4caf50;">{real_count}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="color: white;">🚨 Fake</h4>
                                <h3 style="color: #ff6b6b;">{fake_count}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="color: white;">⚠️ Errors</h4>
                                <h3 style="color: #ffa726;">{error_count}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Show processed data
                        st.markdown("#### 📋 Processed Results")
                        st.dataframe(df, use_container_width=True)
                        
                        # Download button
                        csv_buffer = BytesIO()
                        df.to_csv(csv_buffer, index=False)
                        csv_buffer.seek(0)
                        
                        st.download_button(
                            label="📥 Download Classified Results",
                            data=csv_buffer.getvalue(),
                            file_name=f"classified_news_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
            else:
                st.error("❌ No text columns found in the uploaded CSV file.")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# Footer
st.markdown("""
<div class="footer">
    <p>🤖 <strong>AI-Powered Fake News Detection</strong></p>
    <p>Built with BERT Transformer | Fine-tuned on News Dataset</p>
    <p>⚡ Real-time Analysis & Batch Processing Capabilities</p>
</div>
""", unsafe_allow_html=True)
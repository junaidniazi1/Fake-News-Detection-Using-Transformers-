# 📰 Fake News Detection with BERT



A comprehensive AI-powered fake news detection system built with BERT transformer model. This application provides both single text analysis and batch processing capabilities through an intuitive web interface.

## 🎯 Overview

This project tackles the growing problem of misinformation by leveraging state-of-the-art natural language processing techniques. The system can accurately classify news articles as "REAL" or "FAKE" with confidence scores, making it valuable for journalists, researchers, and social media platforms.

## ✨ Features

- **🔍 Single Text Analysis**: Real-time classification of individual news articles
- **📊 Batch Processing**: Upload CSV files for bulk classification
- **💯 Confidence Scoring**: Get probability scores for prediction reliability
- **📱 Modern UI**: Beautiful, responsive interface with glassmorphism design
- **📥 Export Results**: Download classified results as CSV files
- **⚡ Fast Processing**: Optimized for quick predictions using BERT model

## 🎥 Demo

[![Fake News Detection Demo](https://img.youtube.com/vi/zdFI6_ExDms/0.jpg)](https://youtu.be/zdFI6_ExDms)

**[🎬 Watch Full Project Demo](https://youtu.be/zdFI6_ExDms)**

## 🛠️ Technology Stack

- **Frontend**: Streamlit with custom CSS
- **Model**: BERT (Bidirectional Encoder Representations from Transformers)
- **Framework**: PyTorch, Transformers (Hugging Face)
- **Data Processing**: Pandas, NumPy
- **Language**: Python 3.8+

## 📊 Dataset

The model is trained on the **Fake and Real News Dataset** from Kaggle:
- **Source**: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Size**: 40,000+ news articles
- **Classes**: Real (0) and Fake (1)
- **Features**: Title, Text, Subject, Date

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip package manager
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fake-news-detection-bert.git
cd fake-news-detection-bert
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the trained model** (if not included)
```bash
# Place your trained BERT model in the specified path
# Or download from releases section
```

4. **Run the application**
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## 📁 Project Structure

```
fake-news-detection-bert/
├── app.py                          # Main Streamlit application
├── saved_model/
│   └── fake_news_bert/             # Trained BERT model
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       ├── vocab.txt
│       └── special_tokens_map.json
├── requirements.txt                # Python dependencies
├── README.md                      # Project documentation
├── assets/                        # Images and media files
├── notebooks/                     # Jupyter notebooks for training
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
└── utils/                         # Utility functions
    ├── data_loader.py
    ├── model_utils.py
    └── preprocessing.py
```

## 📋 Requirements

```txt
streamlit>=1.28.0
transformers>=4.30.0
torch>=2.0.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
```

## 🎮 Usage

### Single Text Analysis

1. Navigate to the **"Single Text Analysis"** tab
2. Enter your news text in the text area
3. Click **"Analyze News"** button
4. View the prediction result with confidence score

### Batch Processing

1. Switch to the **"Batch CSV Processing"** tab
2. Upload a CSV file with a text column containing news articles
3. Select the appropriate column name
4. Click **"Process All News"** to analyze all articles
5. Download the results with classifications and confidence scores

### CSV Format

Your CSV file should contain a column with news text:

```csv
text,source,date
"Breaking: New scientific discovery changes everything...","ScienceNews","2024-01-15"
"SHOCKING: Celebrity secrets revealed...","GossipSite","2024-01-16"
```

## 🧠 Model Details

### Architecture
- **Base Model**: BERT (bert-base-uncased)
- **Task**: Binary Classification (Real vs Fake)
- **Input**: Text sequences up to 512 tokens
- **Output**: Probability scores for each class

### Training Details
- **Optimizer**: AdamW
- **Learning Rate**: 2e-5
- **Batch Size**: 16
- **Epochs**: 3-5
- **Loss Function**: Cross Entropy Loss

### Performance Metrics
- **Accuracy**: ~94%
- **Precision**: ~93%
- **Recall**: ~95%
- **F1-Score**: ~94%

## 🔬 Model Training

To train the model from scratch:

1. **Prepare the dataset**
```bash
python utils/data_loader.py
```

2. **Run training script**
```bash
python train.py --epochs 3 --batch_size 16 --learning_rate 2e-5
```

3. **Evaluate the model**
```bash
python evaluate.py --model_path saved_model/fake_news_bert
```

## 📊 Example Predictions

### Real News Example
```
Input: "Apple Inc. announced its quarterly earnings today, reporting revenue of $89.5 billion..."
Output: REAL (98.7% confidence)
```

### Fake News Example
```
Input: "BREAKING: Secret government documents reveal alien technology in underground bunkers..."
Output: FAKE (96.3% confidence)
```

## 🚀 Deployment

### Streamlit Cloud
1. Fork this repository
2. Connect to Streamlit Cloud
3. Deploy with one click

### Docker
```bash
# Build Docker image
docker build -t fake-news-detector .

# Run container
docker run -p 8501:8501 fake-news-detector
```

### Heroku
```bash
# Login to Heroku
heroku login

# Create app
heroku create your-app-name

# Deploy
git push heroku main
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for the BERT implementation
- [Streamlit](https://streamlit.io/) for the amazing web framework
- [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) for providing the dataset
- The open-source community for inspiration and support

## 📧 Contact

**Your Name** - [junaidkhan99e9@gmail.com](mailto:your.email@example.com)



**Demo Video**: [https://youtu.be/zdFI6_ExDms](https://youtu.be/zdFI6_ExDms)

---

⭐ **Star this repository if you found it helpful!**

![GitHub stars](https://img.shields.io/github/stars/yourusername/fake-news-detection-bert?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/fake-news-detection-bert?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/fake-news-detection-bert)
![GitHub license](https://img.shields.io/github/license/yourusername/fake-news-detection-bert)

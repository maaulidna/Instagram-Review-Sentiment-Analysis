# Instagram Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://www.tensorflow.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-latest-F7931E.svg)](https://scikit-learn.org/)

A comprehensive sentiment analysis project that analyzes Instagram app reviews to understand user sentiments. This project implements and compares three different machine learning approaches: **LSTM (Deep Learning)**, **SVM + TF-IDF**, and **Logistic Regression + TF-IDF**.

## 📚 Table of Contents

- [About The Project](#about-the-project)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Models Implemented](#models-implemented)
- [Results & Comparison](#results--comparison)
- [Usage](#usage)
- [Key Findings](#key-findings)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 About The Project

This project performs sentiment analysis on Instagram app reviews scraped from the Google Play Store. The analysis classifies reviews into three categories:
- **Positive (Positif)**: Favorable reviews
- **Neutral (Netral)**: Neutral reviews
- **Negative (Negatif)**: Unfavorable reviews

### Key Features
- Multi-model comparison (LSTM, SVM, Logistic Regression)
- Indonesian language text preprocessing using Sastrawi
- TF-IDF vectorization for traditional ML models
- Word embeddings for deep learning approach
- Comprehensive evaluation metrics

## 📊 Dataset

- **Source**: Instagram app reviews from Google Play Store
- **Dataset URL**: `https://raw.githubusercontent.com/mauliidna/sentiment-sajda/refs/heads/main/igreview.csv`
- **Total Reviews**: 100 samples
- **Features**:
  - `reviewId`: Unique review identifier
  - `userName`: Reviewer's name
  - `content`: Review text (main feature for analysis)
  - `score`: Rating (1-5 stars)
  - `at`: Review timestamp
  - Other metadata fields

### Label Distribution
- **Positive**: Majority class (85% of data)
- **Neutral**: Minority class
- **Negative**: Minority class

## 📁 Project Structure
```

├── File_kode_scraping.ipynb      # Notebook for scraping Instagram comments
├── Pelatihan_Model.ipynb         # Notebook for preprocessing & model training
├── igreview.csv                  # Raw scraped Instagram comments
├── igreview_clean.csv            # Cleaned & preprocessed dataset
├── requirements.txt              # Required dependencies
└── README.md                     # Project documentation

````

## 🛠️ Technologies Used

### Core Libraries

**Data Processing**
- Pandas
- NumPy

**Text Processing**
- Sastrawi (Indonesian NLP)
  - StopWordRemover
  - Stemmer
- Regular Expressions (re)

**Machine Learning**
- Scikit-learn
  - TfidfVectorizer
  - LabelEncoder
  - LinearSVC
  - LogisticRegression
  - Train-test split utilities

**Deep Learning**
- TensorFlow/Keras
  - Sequential Model
  - LSTM layers
  - Embedding layers
  - Dense layers

**Visualization**
- Matplotlib
- Seaborn

## 🚀 Installation

### 1. Clone Repository

```bash
git clone https://github.com/maaulidna/sentiment-ig-analysis.git
cd sentiment-ig-analysis
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn tensorflow scikit-learn Sastrawi tqdm jupyter
```

## 🔄 Data Preprocessing

The preprocessing pipeline includes:

1. **Text Cleaning**
   - Convert to lowercase
   - Remove numbers
   - Remove punctuation
   - Remove extra whitespace

2. **Stopword Removal**
   - Using Sastrawi's Indonesian stopword list

3. **Stemming**
   - Using Sastrawi's Indonesian stemmer
   - Convert words to root form

4. **Label Creation**
   - Score 1-2: Negative
   - Score 3: Neutral
   - Score 4-5: Positive

### Example Preprocessing
```
Original: "pokoknya mantap"
After Cleaning: "pokok mantap"

Original: "lu jangan nampilin ular di beranda gua oon, kalo gua gabisa liat konten, gua bisa copot aplikasi"
After Cleaning: "lu jangan nampilin ular beranda gua oon kalo gua gabisa liat konten gua bisa copot aplikasi"
```

## 🤖 Models Implemented

### 1. LSTM (Long Short-Term Memory) - Deep Learning

**Architecture:**
```
Embedding(10000, 128) → LSTM(128, dropout=0.2) → Dense(64, relu) → Dropout(0.3) → Dense(3, softmax)
```

**Features:**
- Vocabulary size: 10,000
- Embedding dimension: 128
- Sequence length: 100
- Epochs: 5
- Batch size: 64

**Results:**
- **Accuracy**: 85.00%
- Understands context and word order
- Best for complex text structures

### 2. SVM (LinearSVC) + TF-IDF

**Configuration:**
- TF-IDF max features: 5000
- N-gram range: (1, 2)
- Linear kernel

**Results:**
- **Accuracy**: 80.00% (on small test set)
- Fast training
- Stable performance
- Struggles with minority classes

### 3. Logistic Regression + TF-IDF

**Configuration:**
- TF-IDF max features: 5000
- N-gram range: (1, 2)
- Max iterations: 1000

**Results:**
- **Accuracy**: 88.03% ⭐ **BEST MODEL**
- Simple and interpretable
- Fast inference
- Excellent for positive class

## 📈 Results & Comparison

### Model Performance Summary

| Model | Accuracy | Strengths | Weaknesses |
|-------|----------|-----------|------------|
| **LSTM** | 85.00% | • Understands context<br>• Handles sequential data | • Longer training time<br>• More complex |
| **SVM** | 80.00% | • Fast training<br>• Stable | • Poor on minority classes<br>• No context awareness |
| **Logistic Regression** | 88.03% ⭐ | • Highest accuracy<br>• Simple<br>• Fast | • Weak on neutral class<br>• Limited context understanding |

### Detailed Metrics (Logistic Regression)

```
              precision    recall  f1-score   support

     negatif       0.77      0.85      0.80       749
      netral       0.00      0.00      0.00       127
     positif       0.92      0.94      0.93      2124

    accuracy                           0.88      3000
```

### Visualizations

The project includes:
- Score distribution charts
- Text length distribution
- Training accuracy/loss curves (LSTM)
- Model comparison graphs

## 💻 Usage

### Running the Analysis

```python
# 1. Load and preprocess data
import pandas as pd
url = 'https://raw.githubusercontent.com/mauliidna/sentiment-sajda/refs/heads/main/igreview.csv'
df = pd.read_csv(url)

# 2. Apply preprocessing
df['clean_text'] = df['content'].apply(preprocess)

# 3. Train model (e.g., Logistic Regression)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 4. Make predictions
sample = ["Pengalaman saya sangat menyenangkan menggunakan aplikasi ini"]
prediction = model.predict(tfidf.transform(sample))
```

### Example Predictions

```python
# Positive example
text = "aplikasi bagus"
# Output: positif

# Negative example  
text = "pengalaman saya menggunakan aplikasi sangat buruk"
# Output: positif (note: model bias towards positive class)
```

## 🔍 Key Findings

1. **Class Imbalance**: Dataset heavily skewed towards positive reviews (85%)
2. **Best Performer**: Logistic Regression achieved 88.03% accuracy
3. **Neutral Class Challenge**: All models struggled with neutral sentiment
4. **Indonesian NLP**: Sastrawi effectively handles Indonesian text preprocessing
5. **Model Trade-offs**:
   - LSTM: Better context but slower
   - Logistic Regression: Fast and accurate for dominant class
   - SVM: Balanced but limited performance

## 🔮 Future Improvements

- [ ] Address class imbalance using SMOTE or class weights
- [ ] Collect more diverse and balanced dataset
- [ ] Implement ensemble methods
- [ ] Try BERT-based models for Indonesian (IndoBERT)
- [ ] Add aspect-based sentiment analysis
- [ ] Create web API for real-time predictions
- [ ] Develop sentiment dashboard
- [ ] Fine-tune hyperparameters
- [ ] Cross-validation for robust evaluation
- [ ] Add more Indonesian-specific preprocessing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` file for more information.

## 👤 Contact

**Maulidna**

- GitHub: [@maaulidna](https://github.com/maaulidna)
- Project Link: [https://github.com/maaulidna/sentiment-ig-analysis](https://github.com/maaulidna/sentiment-ig-analysis)

## 🙏 Acknowledgments

- [Sastrawi](https://github.com/sastrawi/sastrawi) - Indonesian NLP library
- [Scikit-learn](https://scikit-learn.org/) - Machine learning framework
- [TensorFlow/Keras](https://www.tensorflow.org/) - Deep learning framework
- Google Play Store - Data source

---

⭐ **Star this repository if you find it helpful!**

**#SentimentAnalysis #NLP #Instagram #Python #MachineLearning #DeepLearning #LSTM #TensorFlow #IndonesianNLP**

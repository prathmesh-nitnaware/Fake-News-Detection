# 📰 Fake News Detection using Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=for-the-badge&logo=scikitlearn">
  <img src="https://img.shields.io/badge/Flask-Web%20Application-black?style=for-the-badge&logo=flask">
  <img src="https://img.shields.io/badge/NLP-TF--IDF-green?style=for-the-badge">
</p>

<h1 align="center">Fake News Detection using Machine Learning</h1>

<p align="center">
  <b>Detect Fake and Real News Articles using NLP and Machine Learning</b>
</p>

---

## 📖 Overview

In the digital era, misinformation spreads rapidly through social media and online platforms. Identifying fake news manually is difficult and time-consuming.

This project uses **Natural Language Processing (NLP)** and **Machine Learning** techniques to automatically classify news articles as:

- ✅ Real News
- ❌ Fake News

The application analyzes the textual content of a news article, performs preprocessing, extracts features using **TF-IDF Vectorization**, and predicts whether the news is genuine or misleading.

---

## ✨ Features

- ✅ Fake vs Real News Classification
- ✅ Text Preprocessing and Cleaning
- ✅ TF-IDF Feature Extraction
- ✅ Machine Learning-Based Prediction
- ✅ Real-Time News Classification
- ✅ User-Friendly Web Interface
- ✅ Fast and Lightweight Deployment
- ✅ Easy Integration and Scalability

---

## 🛠️ Technology Stack

### Programming Language
- Python

### Machine Learning Libraries
- Scikit-learn
- Pandas
- NumPy

### Natural Language Processing
- NLTK
- TF-IDF Vectorization

### Web Development
- Flask
- HTML5
- CSS3

---

## 🏗️ Project Architecture

```text
                ┌─────────────────┐
                │   User Input    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Text Processing │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ TF-IDF Features │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Trained ML Model│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Real / Fake News│
                └─────────────────┘
```

---

## 📂 Dataset

The model is trained on a dataset containing thousands of news articles categorized into:

- 🟢 Real News
- 🔴 Fake News

### Dataset Structure

| Column | Description |
|----------|------------|
| title | News Headline |
| text | News Content |
| label | Real/Fake Classification |

---

## 🧹 Data Preprocessing

Before training the model, the following preprocessing steps are performed:

1. Convert text to lowercase
2. Remove punctuation
3. Remove special characters
4. Remove stop words
5. Tokenization
6. Text normalization
7. TF-IDF feature extraction

### Example

#### Input

```text
Breaking News!!! Government announces new policy.
```

#### Processed Output

```text
breaking news government announces new policy
```

---

## 🤖 Machine Learning Workflow

### 1️⃣ Data Collection

News articles are collected and labeled as:

- Real
- Fake

### 2️⃣ Data Cleaning

Removing:

- Noise
- Special characters
- Stopwords
- Unnecessary symbols

### 3️⃣ Feature Extraction

Using **TF-IDF Vectorization** to convert textual data into numerical vectors.

### 4️⃣ Model Training

Training the machine learning classifier using processed data.

### 5️⃣ Model Evaluation

Performance is measured using:

- Accuracy
- Precision
- Recall
- F1 Score

### 6️⃣ Deployment

Deploying the trained model through a Flask web application.

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/prathmesh-nitnaware/Fake-News-Detection.git

cd Fake-News-Detection
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Flask Server:

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## 📖 Usage

### Step 1

Launch the application.

### Step 2

Enter a news article or headline.

### Step 3

Click **Predict**.

### Step 4

View the prediction result.

---

## 💡 Example

### Input

```text
Scientists discover water on Mars.
```

### Output

```text
Prediction: Real News
```

---


## 📁 Project Structure

```text
Fake-News-Detection/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
│
├── dataset/
│   └── news.csv
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── screenshots/
│
├── requirements.txt
├── README.md
│
└── .gitignore
```

---

## 📸 Screenshots

Add screenshots of your application here.

```markdown
![Home Page](screenshots/home.png)

![Prediction Result](screenshots/result.png)
```

---

## 🔮 Future Improvements

- 🚀 Deep Learning Models (LSTM, GRU)
- 🚀 Transformer Models (BERT)
- 🚀 Multi-Language Fake News Detection
- 🚀 News Source Credibility Analysis
- 🚀 REST API Integration
- 🚀 Cloud Deployment
- 🚀 Real-Time News Verification
- 🚀 Browser Extension Support

---

## 🎯 Learning Outcomes

This project helped in understanding:

- Natural Language Processing (NLP)
- Text Classification
- Feature Engineering
- Machine Learning Model Training
- Model Evaluation Techniques
- Flask Web Development
- Model Deployment

---

## 🤝 Contributing

Contributions are welcome!

### Steps to Contribute

1. Fork the repository

2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit changes

```bash
git commit -m "Add new feature"
```

4. Push to GitHub

```bash
git push origin feature-name
```

5. Create a Pull Request

---

## 👨‍💻 Author

### Prathmesh Nitnaware

📧 Passionate about Machine Learning, NLP, and Web Development.

⭐ If you found this project useful, please consider giving it a **Star** on GitHub!

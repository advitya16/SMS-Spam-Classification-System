# 📩 SMS Spam Classification System

An end-to-end Machine Learning and Natural Language Processing (NLP) web application designed to classify messages into **Spam** or **Ham (Legitimate)**. Built following modular software development principles.

---

## 📌 Project Overview

Spam messages reduce productivity and pose security risks. This system processes raw text data, converts text into numerical features using **TF-IDF Vectorization**, and classifies incoming messages using a **Multinomial Naive Bayes** model optimized for high precision.



##  Repository Structure

minor project advitya/
│
├── __pycache__/        # Python compiled cache
├── words.py            # Text preprocessing (lowercasing, stop-words, stemming)
├── mainmodel.py        # Data loading, model training & artifact generation
├── index.py            # Streamlit web application interface
├── pridict.py          # Terminal CLI interface for instant testing
├── spam.csv            # SMS Spam Collection Dataset
├── vectorizer.pkl      # Saved TF-IDF Vectorizer
└── model.pkl           # Saved Naive Bayes Machine Learning Model


## Tech Stack


Language: Python
Data Manipulation: Pandas, NumPy
NLP Pipeline: NLTK
Machine Learning: Scikit-Learn (Multinomial Naive Bayes, TF-IDF Vectorizer)
Web Interface: Streamlit
Model Persistence: Pickle



# How It Works

Text Cleaning (words.py)
Lowercases raw input text.
Tokenizes text into individual words.
Filters out non-alphanumeric characters, punctuation, and English stop-words.
Applies Porter Stemming to standardize words to root form.
Model Training (mainmodel.py)
Cleans dataset and encodes binary targets (ham -> 0, spam -> 1).
Extracts top 3,000 numerical features using TF-IDF Vectorization.
Trains Multinomial Naive Bayes classifier.
Exports vectorizer.pkl and model.pkl.
Web Application (index.py)
Interactive Streamlit dashboard.

Takes raw user input, transforms text on the fly, and displays classification along with confidence percentage.

Terminal Interface (pridict.py)

Command-line script to perform quick predictions without starting the web application.

## How to Run
1. Install Dependencies
Bash
pip install pandas numpy scikit-learn nltk streamlit
2. Train Model & Generate Artifacts
Bash
python mainmodel.py
3. Run Streamlit Web Application
Bash
python -m streamlit run index.py
4. Run Terminal Predictor (Optional)
Bash
python pridict.py
## Model Evaluation Metrics
Accuracy: ~93%


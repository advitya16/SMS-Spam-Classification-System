import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from words import clean_words

def train_and_save_model():
    print("1. Loading dataset...")
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')
    df.columns = ['target', 'text']
    df['target'] = df['target'].map({'ham': 0, 'spam': 1})
    df = df.drop_duplicates(keep='first').reset_index(drop=True)
    
    print("2. Processing text using words.py...")
    df['transformed_text'] = df['text'].apply(clean_words)
    
    print("3. Vectorizing text with TF-IDF...")
    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df['transformed_text']).toarray()
    y = df['target'].values
    
    print("4. Training Multinomial Naive Bayes model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("\n--- Model Training Results ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")
    
    print("5. Saving model artifacts...")
    pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
    pickle.dump(model, open('model.pkl', 'wb'))
    print("Done! Saved 'vectorizer.pkl' and 'model.pkl'.")

if __name__ == '__main__':
    train_and_save_model()
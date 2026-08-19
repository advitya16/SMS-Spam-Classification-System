import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()

def clean_words(text):
    text = str(text).lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]
    stemmed = [ps.stem(t) for t in tokens]
    return " ".join(stemmed)
import re
import contractions
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def expand_contractions(text):
    """Espande le contrazioni come "can't" -> "cannot""" 
    return contractions.fix(text)

def clean_content(text):
    """Esegue la pulizia del testo rimuovendo handle Twitter, link, punteggiatura, stop words e lemmatizzando le parole."""
    text = expand_contractions(text)
    text = re.sub(r'@\w+\s?', '', text)  # Rimuove handle Twitter
    test = text.lower()
    text = re.sub(r'https?:\/\/\S+', '', text)  # Rimuove link http
    text = re.sub(r'www\.[a-z]?\.?com|[a-z]+\.com', '', text)  # Rimuove link www
    text = re.sub(r'&[a-z]+;', '', text)  # Rimuove riferimenti HTML
    text = re.sub(r"[^a-z\s\(\-:\)\\\/\];='#]", '', text)  # Mantiene solo lettere e spazi
    text = text.split()
    
    clean_lst = [word for word in text if word not in stop_words]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in clean_lst]
    
    return ' '.join(lemmatized_words)
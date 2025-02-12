import string
import nltk
from nltk.tokenize import TweetTokenizer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

tokenizer = TweetTokenizer()
PUNCTUATION_LIST = set(string.punctuation)

def tokenize_text(text):
    """Tokenizza il testo usando il tokenizer di Twitter."""
    return tokenizer.tokenize(text)

def remove_punctuation(word_list):
    """Rimuove i segni di punteggiatura da una lista di parole."""
    return [w for w in word_list if w not in PUNCTUATION_LIST]
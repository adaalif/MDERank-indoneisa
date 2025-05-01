from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re

# Initialize Indonesian stopwords
factory = StopWordRemoverFactory()
stopword_dict = set(factory.get_stop_words())

# Initialize Indonesian stemmer
stemmer = StemmerFactory().create_stemmer()

# Indonesian grammar patterns for noun phrases
# Adapted for Indonesian grammar structure
GRAMMAR_ID = """  NP:
        {<NN.*|JJ>*<NN.*>}  # Kata sifat(opsional) + Kata benda
        {<VB.*>?<NN.*>+}    # Kata kerja(opsional) + Kata benda
        {<NN.*>+}           # Rangkaian kata benda
"""

def preprocess_indonesian_text(text):
    """
    Preprocess Indonesian text:
    - Convert to lowercase
    - Remove special characters
    - Remove extra whitespace
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_indonesian_candidates(text, pos_tagger):
    """
    Extract candidate keyphrases from Indonesian text
    Args:
        text: Input text
        pos_tagger: POS tagger function
    Returns:
        List of candidate keyphrases
    """
    # Preprocess text
    text = preprocess_indonesian_text(text)
    
    # Split into tokens
    tokens = text.split()
    
    # Remove stopwords
    tokens = [t for t in tokens if t not in stopword_dict]
    
    # Extract noun phrases (simplified version)
    candidates = []
    i = 0
    while i < len(tokens):
        # Single word candidates
        candidates.append(tokens[i])
        
        # Two word candidates
        if i + 1 < len(tokens):
            candidates.append(f"{tokens[i]} {tokens[i+1]}")
            
        # Three word candidates
        if i + 2 < len(tokens):
            candidates.append(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")
        
        i += 1
    
    return candidates

def stem_indonesian_phrase(phrase):
    """
    Stem an Indonesian phrase using Sastrawi
    """
    return " ".join(stemmer.stem(word) for word in phrase.split()) 
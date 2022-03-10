"""
pre-process given text
"""
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List
from gensim.models import Phrases
import numpy as np

wordnet_lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
stop_words = stopwords.words('english')
nltk.download('wordnet')


filters : List[str] = ['NOUN', 'ADJ', 'PROPN', 'VERB', 'ADV']

def cleanse(text: str)-> List[str]:
    """
    pre-process (lowercase and remove non-alphabets); tokenize; lemmatize; remove stop words; for each text example.
    """
    text = text.lower()          # bring all words to same case: lowercase
    text = re.sub('\s+', ' ', re.sub("[^a-z]+", ' ', text)).strip()    #remove non-alphabetic characters 
    text = text.split()           # tokenize the paragraph into words
    words = [wordnet_lemmatizer.lemmatize(word, 'v') for word in text if word not in stop_words]    #lemmatization
    return words

def bigram_features(common_texts: np.ndarray) -> np.ndarray:
    """
    identify commonly occuring bigrams
    """
    bigrams = Phrases(common_texts, min_count=5)
    for idx in range(len(common_texts)):
        for token in bigrams[common_texts[idx]]:
            if '_' in token:
                common_texts[idx].append(token)
    return common_texts
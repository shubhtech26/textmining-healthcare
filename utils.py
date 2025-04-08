import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
import os

# Define data directory path
DATA_DIR = 'data'

def load_clinical_stopwords(filepath='clinical-stopwords.txt'):
    """Load clinical stopwords from file."""
    filepath = os.path.join(DATA_DIR, filepath)
    with open(filepath, 'r') as f:
        clinical_stopwords = set([line.strip() for line in f])
    return clinical_stopwords

def load_classes(filepath='classes.txt'):
    """Load class labels from file."""
    filepath = os.path.join(DATA_DIR, filepath)
    with open(filepath, 'r') as f:
        classes = [line.strip() for line in f]
    return classes

def preprocess_text(text, clinical_stopwords=None):
    """Preprocess text by removing special characters, numbers, and stopwords."""
    if clinical_stopwords is None:
        clinical_stopwords = load_clinical_stopwords()
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Simple word tokenization
    tokens = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english')).union(clinical_stopwords)
    tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)

def load_and_preprocess_data(train_path='train.csv', test_path='test.csv'):
    """Load and preprocess training and test data."""
    train_path = os.path.join(DATA_DIR, train_path)
    test_path = os.path.join(DATA_DIR, test_path)
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    clinical_stopwords = load_clinical_stopwords()
    
    # Preprocess text data
    print("Preprocessing training data...")
    train_df['processed_text'] = train_df['text'].apply(
        lambda x: preprocess_text(x, clinical_stopwords))
    
    print("Preprocessing test data...")
    test_df['processed_text'] = test_df['text'].apply(
        lambda x: preprocess_text(x, clinical_stopwords))
    
    return train_df, test_df 
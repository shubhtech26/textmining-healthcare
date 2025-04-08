import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import numpy as np

def plot_class_distribution(df, class_column='label'):
    """Plot the distribution of classes in the dataset."""
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=class_column)
    plt.xticks(rotation=45)
    plt.title('Distribution of Medical Document Classes')
    plt.tight_layout()
    return plt

def create_wordcloud(texts, title='Word Cloud of Medical Documents'):
    """Create and plot a word cloud from the text data."""
    plt.figure(figsize=(12, 8))
    wordcloud = WordCloud(width=800, height=400,
                         background_color='white',
                         max_words=200).generate(' '.join(texts))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    return plt

def plot_document_lengths(df, text_column='processed_text', class_column='label'):
    """Plot the distribution of document lengths by class."""
    df['doc_length'] = df[text_column].str.split().str.len()
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x=class_column, y='doc_length')
    plt.xticks(rotation=45)
    plt.title('Document Length Distribution by Class')
    plt.ylabel('Number of Words')
    plt.tight_layout()
    return plt

def plot_top_terms_by_class(df, text_column='processed_text', 
                           class_column='label', n_terms=10):
    """Plot top n terms for each class."""
    classes = df[class_column].unique()
    fig, axes = plt.subplots(len(classes), 1, figsize=(12, 4*len(classes)))
    
    for idx, class_name in enumerate(classes):
        class_texts = ' '.join(df[df[class_column] == class_name][text_column])
        words = class_texts.split()
        word_freq = Counter(words).most_common(n_terms)
        
        words, freqs = zip(*word_freq)
        sns.barplot(x=list(freqs), y=list(words), ax=axes[idx])
        axes[idx].set_title(f'Top {n_terms} Terms in {class_name}')
        axes[idx].set_xlabel('Frequency')
    
    plt.tight_layout()
    return plt

def plot_length_distribution(df, text_column='processed_text'):
    """Plot the distribution of document lengths."""
    df['doc_length'] = df[text_column].str.split().str.len()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='doc_length', bins=50)
    plt.title('Distribution of Document Lengths')
    plt.xlabel('Number of Words')
    plt.ylabel('Count')
    return plt 
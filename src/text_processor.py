"""
Text processing utilities for resume screening.

This module provides comprehensive text preprocessing capabilities including
cleaning, stopwords removal, lemmatization, and vectorization.
"""

import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TextProcessor:
    """
    A comprehensive text processor for resume data.
    
    Handles text cleaning, NLP processing, and feature extraction
    for both traditional ML and deep learning approaches.
    """
    
    def __init__(self):
        """Initialize the text processor with necessary components."""
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_vectorizer = None
        self.tokenizer = None
        self.max_sequence_length = 500
        self.max_vocab_size = 10000
        
    def clean_text(self, text):
        """
        Clean and preprocess text data.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Remove URLs
        text = re.sub(r'http\S+\s*', ' ', text)
        
        # Remove RT and cc
        text = re.sub(r'RT|cc', ' ', text)
        
        # Remove hashtags
        text = re.sub(r'#\S+', '', text)
        
        # Remove mentions
        text = re.sub(r'@\S+', ' ', text)
        
        # Remove punctuation
        text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
        
        # Remove non-ASCII characters
        text = re.sub(r'[^\x00-\x7f]', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.lower().strip()
    
    def nlp_process(self, text):
        """
        Apply NLP processing including stopwords removal and lemmatization.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Processed text
        """
        if not isinstance(text, str):
            return ""
            
        words = text.split()
        
        # Remove stopwords
        words = [word for word in words if word not in self.stop_words]
        
        # Lemmatize
        words = [self.lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(words)
    
    def process_dataframe(self, df, text_column='Resume'):
        """
        Process a DataFrame containing resume text.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            text_column (str): Name of the text column
            
        Returns:
            pd.DataFrame: Processed DataFrame with additional columns
        """
        df_processed = df.copy()
        
        # Clean text
        df_processed['cleaned_resume'] = df_processed[text_column].apply(self.clean_text)
        
        # Apply NLP processing
        df_processed['processed_resume'] = df_processed['cleaned_resume'].apply(self.nlp_process)
        
        return df_processed
    
    def fit_tfidf_vectorizer(self, texts):
        """
        Fit TF-IDF vectorizer on training texts.
        
        Args:
            texts (list): List of processed texts
        """
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.tfidf_vectorizer.fit(texts)
    
    def transform_tfidf(self, texts):
        """
        Transform texts using fitted TF-IDF vectorizer.
        
        Args:
            texts (list): List of processed texts
            
        Returns:
            scipy.sparse matrix: TF-IDF features
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf_vectorizer first.")
        
        return self.tfidf_vectorizer.transform(texts)
    
    def fit_tokenizer(self, texts):
        """
        Fit Keras tokenizer for deep learning models.
        
        Args:
            texts (list): List of processed texts
        """
        self.tokenizer = Tokenizer(
            num_words=self.max_vocab_size,
            oov_token='<UNK>'
        )
        self.tokenizer.fit_on_texts(texts)
    
    def texts_to_sequences(self, texts):
        """
        Convert texts to padded sequences for deep learning.
        
        Args:
            texts (list): List of processed texts
            
        Returns:
            np.array: Padded sequences
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not fitted. Call fit_tokenizer first.")
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_sequence_length,
            padding='post',
            truncating='post'
        )
        
        return padded_sequences
    
    def get_vocabulary_size(self):
        """
        Get the size of the vocabulary.
        
        Returns:
            int: Vocabulary size
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not fitted.")
        
        return len(self.tokenizer.word_index) + 1
    
    def preprocess_single_resume(self, resume_text):
        """
        Preprocess a single resume text for prediction.
        
        Args:
            resume_text (str): Raw resume text
            
        Returns:
            dict: Processed features for both ML and DL approaches
        """
        # Clean and process text
        cleaned = self.clean_text(resume_text)
        processed = self.nlp_process(cleaned)
        
        result = {
            'raw_text': resume_text,
            'cleaned_text': cleaned,
            'processed_text': processed
        }
        
        # Add TF-IDF features if vectorizer is fitted
        if self.tfidf_vectorizer is not None:
            result['tfidf_features'] = self.transform_tfidf([processed])
        
        # Add sequence features if tokenizer is fitted
        if self.tokenizer is not None:
            result['sequence_features'] = self.texts_to_sequences([processed])
        
        return result

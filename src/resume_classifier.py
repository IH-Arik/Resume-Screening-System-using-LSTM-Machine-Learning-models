"""
Main Resume Classifier interface.

This module provides a unified interface for resume classification
using multiple trained models and preprocessing utilities.
"""

import pickle
import joblib
import numpy as np
from .text_processor import TextProcessor
from .model_trainer import ModelTrainer


class ResumeClassifier:
    """
    A comprehensive resume classifier that combines multiple models
    and preprocessing utilities for automatic resume categorization.
    """
    
    def __init__(self):
        """Initialize the resume classifier."""
        self.text_processor = TextProcessor()
        self.model_trainer = ModelTrainer()
        self.is_fitted = False
        
    def fit(self, df, text_column='Resume', target_column='Category'):
        """
        Train the classifier on resume data.
        
        Args:
            df (pd.DataFrame): Training data
            text_column (str): Name of the text column
            target_column (str): Name of the target column
        """
        # Process text
        processed_df = self.text_processor.process_dataframe(df, text_column)
        
        # Prepare features
        texts = processed_df['processed_resume'].tolist()
        labels = df[target_column].tolist()
        
        # Fit TF-IDF vectorizer
        self.text_processor.fit_tfidf_vectorizer(texts)
        X_tfidf = self.text_processor.transform_tfidf(texts)
        
        # Fit tokenizer for deep learning
        self.text_processor.fit_tokenizer(texts)
        X_sequences = self.text_processor.texts_to_sequences(texts)
        
        # Prepare data for training
        X_train, X_test, y_train, y_test, _, _ = self.model_trainer.prepare_data(
            X_tfidf, labels
        )
        
        # Train traditional ML models
        self.model_trainer.train_lightgbm(X_train, y_train, X_test, y_test)
        self.model_trainer.train_xgboost(X_train, y_train, X_test, y_test)
        self.model_trainer.train_catboost(X_train, y_train, X_test, y_test)
        self.model_trainer.train_ensemble(X_train, y_train, X_test, y_test)
        
        # Train LSTM model
        X_train_seq, X_test_seq, y_train_seq, y_test_seq, _, _ = self.model_trainer.prepare_data(
            X_sequences, labels
        )
        
        vocab_size = self.text_processor.get_vocabulary_size()
        max_seq_len = self.text_processor.max_sequence_length
        
        self.model_trainer.train_lstm(
            X_train_seq, y_train_seq, X_test_seq, y_test_seq,
            vocab_size, max_seq_len
        )
        
        self.is_fitted = True
        
    def predict(self, resume_text, model_name='ensemble'):
        """
        Predict category for a single resume.
        
        Args:
            resume_text (str): Resume text to classify
            model_name (str): Model to use for prediction
            
        Returns:
            dict: Prediction results with category and confidence
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        # Process resume
        processed = self.text_processor.preprocess_single_resume(resume_text)
        
        # Get features for the selected model
        if model_name == 'lstm':
            features = processed['sequence_features']
        else:
            features = processed['tfidf_features']
        
        # Make prediction
        category, confidence = self.model_trainer.predict_category(model_name, features)
        
        return {
            'category': category,
            'confidence': confidence,
            'processed_text': processed['processed_text'],
            'model_used': model_name
        }
    
    def predict_batch(self, resume_texts, model_name='ensemble'):
        """
        Predict categories for multiple resumes.
        
        Args:
            resume_texts (list): List of resume texts
            model_name (str): Model to use for prediction
            
        Returns:
            list: List of prediction results
        """
        results = []
        for text in resume_texts:
            result = self.predict(text, model_name)
            results.append(result)
        
        return results
    
    def get_model_performance(self):
        """
        Get performance metrics for all trained models.
        
        Returns:
            dict: Performance metrics for each model
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        performance = {}
        for model_name in ['lightgbm', 'xgboost', 'catboost', 'ensemble', 'lstm']:
            model = self.model_trainer.get_model(model_name)
            if model is not None:
                # Note: This would require test data to compute actual metrics
                # For now, return placeholder
                performance[model_name] = {
                    'status': 'trained',
                    'type': 'traditional_ml' if model_name != 'lstm' else 'deep_learning'
                }
        
        return performance
    
    def save_models(self, save_dir='models'):
        """
        Save all trained models and preprocessors.
        
        Args:
            save_dir (str): Directory to save models
        """
        if not self.is_fitted:
            raise ValueError("No models to save. Fit the classifier first.")
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save text processor
        joblib.dump(self.text_processor, f'{save_dir}/text_processor.pkl')
        
        # Save model trainer
        joblib.dump(self.model_trainer, f'{save_dir}/model_trainer.pkl')
        
        # Save individual models
        for model_name in ['lightgbm', 'xgboost', 'catboost', 'ensemble']:
            model = self.model_trainer.get_model(model_name)
            if model is not None:
                joblib.dump(model, f'{save_dir}/{model_name}.pkl')
        
        # Save LSTM model separately
        lstm_model = self.model_trainer.get_model('lstm')
        if lstm_model is not None:
            lstm_model.save(f'{save_dir}/lstm_model.h5')
        
        print(f"Models saved to {save_dir}")
    
    def load_models(self, save_dir='models'):
        """
        Load saved models and preprocessors.
        
        Args:
            save_dir (str): Directory containing saved models
        """
        import os
        
        if not os.path.exists(save_dir):
            raise ValueError(f"Model directory {save_dir} does not exist.")
        
        # Load text processor
        self.text_processor = joblib.load(f'{save_dir}/text_processor.pkl')
        
        # Load model trainer
        self.model_trainer = joblib.load(f'{save_dir}/model_trainer.pkl')
        
        # Load individual models
        for model_name in ['lightgbm', 'xgboost', 'catboost', 'ensemble']:
            model_path = f'{save_dir}/{model_name}.pkl'
            if os.path.exists(model_path):
                self.model_trainer.models[model_name] = joblib.load(model_path)
        
        # Load LSTM model
        if os.path.exists(f'{save_dir}/lstm_model.h5'):
            from tensorflow.keras.models import load_model
            self.model_trainer.models['lstm'] = load_model(f'{save_dir}/lstm_model.h5')
        
        self.is_fitted = True
        print(f"Models loaded from {save_dir}")
    
    def get_categories(self):
        """
        Get list of all categories the classifier can predict.
        
        Returns:
            list: List of category names
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        return self.model_trainer.label_encoder.classes_.tolist()
    
    def get_feature_importance(self, model_name='lightgbm', top_n=20):
        """
        Get feature importance for tree-based models.
        
        Args:
            model_name (str): Name of the model
            top_n (int): Number of top features to return
            
        Returns:
            dict: Feature names and their importance scores
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        model = self.model_trainer.get_model(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found.")
        
        if hasattr(model, 'feature_importances_'):
            feature_names = self.text_processor.tfidf_vectorizer.get_feature_names_out()
            importances = model.feature_importances_
            
            # Get top features
            indices = np.argsort(importances)[::-1][:top_n]
            
            return {
                feature_names[i]: importances[i] 
                for i in indices
            }
        else:
            raise ValueError(f"Model '{model_name}' does not support feature importance.")
    
    def __repr__(self):
        """String representation of the classifier."""
        status = "fitted" if self.is_fitted else "not fitted"
        models = list(self.model_trainer.models.keys()) if self.is_fitted else []
        
        return f"ResumeClassifier(status={status}, models={models})"

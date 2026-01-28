"""
Model training utilities for resume classification.

This module provides training capabilities for various machine learning
models including traditional ML algorithms and deep learning approaches.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns


class ModelTrainer:
    """
    A comprehensive model trainer for resume classification.
    
    Supports training of multiple model types including traditional ML
    algorithms and deep learning approaches.
    """
    
    def __init__(self):
        """Initialize the model trainer."""
        self.models = {}
        self.label_encoder = LabelEncoder()
        self.class_weights = None
        
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """
        Prepare data for training.
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size (float): Proportion of test data
            random_state (int): Random seed
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded)
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        # Calculate class weights for imbalanced datasets
        self.class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        self.class_weights = dict(enumerate(self.class_weights))
        
        return X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded
    
    def train_lightgbm(self, X_train, y_train, X_test, y_test):
        """
        Train LightGBM classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Training results and metrics
        """
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=len(np.unique(y_train)),
            random_state=42,
            class_weight=self.class_weights
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        metrics = self._calculate_metrics(y_test, y_pred, y_prob)
        
        self.models['lightgbm'] = model
        
        return {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_prob,
            'metrics': metrics
        }
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """
        Train XGBoost classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Training results and metrics
        """
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=len(np.unique(y_train)),
            random_state=42,
            eval_metric='mlogloss'
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        metrics = self._calculate_metrics(y_test, y_pred, y_prob)
        
        self.models['xgboost'] = model
        
        return {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_prob,
            'metrics': metrics
        }
    
    def train_catboost(self, X_train, y_train, X_test, y_test):
        """
        Train CatBoost classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Training results and metrics
        """
        model = CatBoostClassifier(
            iterations=100,
            random_state=42,
            verbose=False,
            loss_function='MultiClass'
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        metrics = self._calculate_metrics(y_test, y_pred, y_prob)
        
        self.models['catboost'] = model
        
        return {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_prob,
            'metrics': metrics
        }
    
    def train_ensemble(self, X_train, y_train, X_test, y_test):
        """
        Train ensemble stacking classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Training results and metrics
        """
        # Base estimators
        base_estimators = [
            ('lgb', lgb.LGBMClassifier(random_state=42)),
            ('xgb', xgb.XGBClassifier(random_state=42)),
            ('cat', CatBoostClassifier(iterations=50, random_state=42, verbose=False))
        ]
        
        # Meta-learner
        meta_learner = LogisticRegression(max_iter=1000, random_state=42)
        
        # Stacking classifier
        model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=5
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        metrics = self._calculate_metrics(y_test, y_pred, y_prob)
        
        self.models['ensemble'] = model
        
        return {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_prob,
            'metrics': metrics
        }
    
    def build_lstm_model(self, vocab_size, max_sequence_length, num_classes):
        """
        Build LSTM model for deep learning approach.
        
        Args:
            vocab_size (int): Size of vocabulary
            max_sequence_length (int): Maximum sequence length
            num_classes (int): Number of output classes
            
        Returns:
            tensorflow.keras.Model: Compiled LSTM model
        """
        model = Sequential([
            Embedding(vocab_size, 128, input_length=max_sequence_length),
            LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
            LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_lstm(self, X_train, y_train, X_test, y_test, vocab_size, max_sequence_length):
        """
        Train LSTM model.
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            X_test: Test sequences
            y_test: Test labels
            vocab_size (int): Size of vocabulary
            max_sequence_length (int): Maximum sequence length
            
        Returns:
            dict: Training results and metrics
        """
        # Convert labels to categorical
        y_train_cat = to_categorical(y_train)
        y_test_cat = to_categorical(y_test)
        
        # Build model
        model = self.build_lstm_model(vocab_size, max_sequence_length, len(np.unique(y_train)))
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train, y_train_cat,
            validation_data=(X_test, y_test_cat),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Predictions
        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_prob)
        
        self.models['lstm'] = model
        
        return {
            'model': model,
            'history': history,
            'predictions': y_pred,
            'probabilities': y_pred_prob,
            'metrics': metrics
        }
    
    def _calculate_metrics(self, y_true, y_pred, y_prob):
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            
        Returns:
            dict: Dictionary of metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'auc': roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred)
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, title='Confusion Matrix'):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title (str): Plot title
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def get_model(self, model_name):
        """
        Get a trained model by name.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Trained model or None if not found
        """
        return self.models.get(model_name)
    
    def get_label_encoder(self):
        """
        Get the fitted label encoder.
        
        Returns:
            LabelEncoder: Fitted label encoder
        """
        return self.label_encoder
    
    def predict_category(self, model_name, features):
        """
        Predict category using a trained model.
        
        Args:
            model_name (str): Name of the model
            features: Input features
            
        Returns:
            tuple: (predicted_category, confidence_score)
        """
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found.")
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)
            pred_idx = np.argmax(proba, axis=1)[0]
            confidence = proba.max()
        else:
            pred_idx = model.predict(features)[0]
            confidence = 1.0
        
        category = self.label_encoder.inverse_transform([pred_idx])[0]
        
        return category, confidence

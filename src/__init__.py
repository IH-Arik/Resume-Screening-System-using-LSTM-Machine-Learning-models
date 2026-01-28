"""
Resume Screening with LSTM and Ensemble Methods

A comprehensive machine learning package for automatic resume classification
using advanced NLP techniques, LSTM neural networks, and ensemble learning.
"""

__version__ = "1.0.0"
__author__ = "Resume Screening Team"
__email__ = "contact@example.com"

from .text_processor import TextProcessor
from .model_trainer import ModelTrainer
from .resume_classifier import ResumeClassifier

__all__ = [
    "TextProcessor",
    "ModelTrainer", 
    "ResumeClassifier"
]

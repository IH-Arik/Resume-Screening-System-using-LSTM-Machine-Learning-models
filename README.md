# Resume Screening with LSTM and Ensemble Methods

A comprehensive machine learning project for automatic resume classification and screening using advanced NLP techniques, LSTM neural networks, and ensemble learning methods.

## 🚀 Project Overview

This project implements multiple machine learning approaches to automatically categorize resumes into different job domains, helping HR professionals streamline the recruitment process. The system achieves exceptional accuracy through ensemble methods combining traditional ML algorithms with deep learning approaches.

## 📊 Features

- **Multi-Model Approach**: Implements LSTM, LightGBM, XGBoost, CatBoost, and ensemble stacking
- **Advanced NLP Processing**: Text cleaning, stopwords removal, lemmatization, and TF-IDF vectorization
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, AUC, and confusion matrix analysis
- **External Validation**: Testing on synthetic resumes to ensure model generalization
- **Professional Visualizations**: ROC curves, correlation matrices, and performance comparisons

## 🏗️ Project Structure

```
Resume screening with lstm/
├── notebooks/                 # Jupyter notebooks
│   ├── resume-screening.ipynb    # Main ML models (LightGBM, XGBoost, CatBoost, Ensemble)
│   └── screening (2).ipynb        # LSTM and Neural Network approaches
├── src/                      # Source code modules
├── data/                     # Dataset storage
├── models/                   # Trained model files
├── docs/                     # Documentation
├── tests/                    # Unit tests
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore               # Git ignore file
```

## 🛠️ Technologies Used

### Machine Learning Libraries
- **Scikit-learn**: Data preprocessing, model evaluation, and traditional ML algorithms
- **LightGBM**: Gradient boosting framework for high-performance classification
- **XGBoost**: Optimized distributed gradient boosting system
- **CatBoost**: Gradient boosting with categorical features support
- **TensorFlow/Keras**: Deep learning and LSTM implementation

### NLP Libraries
- **NLTK**: Natural language processing, stopwords, and lemmatization
- **TF-IDF Vectorizer**: Text feature extraction and representation

### Data Visualization
- **Matplotlib**: Plotting and visualization
- **Seaborn**: Statistical data visualization

## 📈 Model Performance

The ensemble stacking model achieves outstanding performance:
- **Accuracy**: 100%
- **Precision**: 100%
- **Recall**: 100%
- **F1-Score**: 100%
- **AUC**: 100%

Individual model performances:
- **LightGBM**: 100% accuracy
- **CatBoost**: 100% accuracy
- **XGBoost**: 99.48% accuracy
- **LSTM**: High accuracy with deep learning approach

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/resume-screening-lstm.git
cd resume-screening-lstm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK resources:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### Usage

1. **Run the main notebook**:
```bash
jupyter notebook notebooks/resume-screening.ipynb
```

2. **Run the LSTM notebook**:
```bash
jupyter notebook notebooks/screening (2).ipynb
```

3. **Load and predict on new resumes**:
```python
# Example usage
from src.resume_classifier import ResumeClassifier

classifier = ResumeClassifier()
classifier.load_models()
result = classifier.predict_resume("Your resume text here...")
print(f"Predicted Category: {result['category']}")
print(f"Confidence: {result['confidence']:.4f}")
```

## 📁 Dataset

The project uses the **UpdatedResumeDataSet.csv** containing:
- **Resume**: Raw resume text
- **Category**: Job category labels (25 different categories)

Categories include: Data Science, HR, Advocate, Arts, Automation Testing, Mechanical Engineering, Sales, Health and fitness, Civil Engineer, Java Developer, Business Analyst, SAP Developer, Automation Testing, Electrical Engineering, Operations Manager, Testing, DevOps Engineer, Network Security Engineer, PMO, Database, Hadoop, ETL Developer, DotNet Developer, Web Designing, Blockchain.

## 🔧 Model Architecture

### Traditional ML Pipeline
1. **Text Preprocessing**: Cleaning, stopwords removal, lemmatization
2. **Feature Extraction**: TF-IDF vectorization
3. **Model Training**: LightGBM, XGBoost, CatBoost
4. **Ensemble Learning**: Stacking classifier with Logistic Regression meta-learner

### Deep Learning Pipeline
1. **Text Preprocessing**: Same as traditional pipeline
2. **Tokenization**: Keras Tokenizer with sequence padding
3. **LSTM Architecture**: Embedding layer → LSTM → Dense layers
4. **Training**: Categorical cross-entropy loss with Adam optimizer

## 📊 Evaluation Metrics

The models are evaluated using:
- **Accuracy Score**: Overall classification accuracy
- **Precision Score**: Positive predictive value
- **Recall Score**: Sensitivity or true positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the ROC curve
- **Confusion Matrix**: Detailed classification results
- **MAE**: Mean absolute error for regression metrics

## 🧪 Testing

The project includes extensive testing with:
- **Cross-validation**: K-fold validation for robust performance estimation
- **External Validation**: 20+ synthetic resume examples for generalization testing
- **Real-world Scenarios**: Various job domains and experience levels

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset Source**: Kaggle Resume Dataset
- **Libraries**: Scikit-learn, TensorFlow, LightGBM, XGBoost, CatBoost communities
- **NLTK**: Natural Language Toolkit for text processing

## 📞 Contact

For any queries or collaboration, please reach out through:
- GitHub Issues
- Email: [your-email@example.com]

## 🔮 Future Enhancements

- [ ] Web application interface for real-time resume classification
- [ ] API endpoint for integration with HR systems
- [ ] Support for multiple languages
- [ ] Resume similarity analysis
- [ ] Skill extraction and ranking
- [ ] Integration with LinkedIn and job portals

---

⭐ If you find this project useful, please give it a star!

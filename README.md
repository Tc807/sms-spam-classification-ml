# sms-spam-classification-ml
AI-based SMS Spam Classifier using NLP &amp; Machine Learning
SMS Spam Classifier

This project is an AI-based SMS Spam Classification system developed as part of an Artificial Intelligence / Machine Learning academic course.

The system automatically classifies SMS messages into two categories:

Ham (legitimate messages)
Spam (unwanted or promotional messages)

Overview

The goal of this project is to build a reliable text classification model using Natural Language Processing (NLP) techniques and deploy it through an interactive web interface. The system processes raw SMS text and predicts whether it is spam or not in real time.

Machine Learning Methodology :

This project follows a complete NLP pipeline:

Text preprocessing using NLTK
Tokenization, stopword removal, and lemmatization
Feature extraction using TF-IDF vectorization
Model training using the Naive Bayes algorithm
Deployment using Streamlit for an interactive user interface

Technologies Used :
Python
Scikit-learn
NLTK
Pandas
NumPy
Streamlit

SMS-Spam-Classifier/
├── app.py              # Streamlit web application
├── train.py            # Model training script
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
├── models/
│   ├── tfidf.pkl       # TF-IDF vectorizer
│   └── nb_model.pkl    # Trained Naive Bayes model

How to Run the Project
1. Install dependencies
pip install -r requirements.txt
2. Run the application
streamlit run app.py
Output
The user inputs an SMS message
The system predicts whether the message is:
Spam
Ham (Not Spam)
Model Training (train.py)

The train.py script performs the following steps:

Loads the SMS Spam dataset
Cleans and preprocesses text data
Converts text into numerical features using TF-IDF
Trains a Naive Bayes classifier
Evaluates performance using accuracy, precision, recall, and F1-score
Saves the trained model and vectorizer as .pkl files
Dataset

Dataset used in this project:

SMS Spam Collection Dataset
UCI Machine Learning Repository
https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

Note: The dataset is not included in the repository to keep it lightweight.

Project Objective

The objective of this project is to demonstrate the practical application of machine learning and NLP techniques in solving real-world text classification problems such as spam detection.

Author

Tushar Choudhary
MCA Student

License

This project is licensed under the MIT License.


Future Improvements :

This project can be further enhanced in the following ways:

1. Use Advanced Models
Replace Naive Bayes with more powerful models such as Logistic Regression, Random Forest, or Gradient Boosting
Experiment with deep learning models like LSTM or Transformer-based architectures (e.g., BERT) for better contextual understanding
2. Improve Text Representation
Move beyond TF-IDF and use word embeddings such as Word2Vec, GloVe, or contextual embeddings like BERT
Capture semantic meaning instead of relying only on word frequency
3. Hyperparameter Tuning
Apply techniques like Grid Search or Random Search to optimize model performance
Fine-tune parameters for better accuracy and generalization
4. Handle Imbalanced Data
Apply techniques such as oversampling (SMOTE) or undersampling
Improve detection of minority class (spam messages)
5. Real-Time Deployment
Deploy the model using cloud platforms such as AWS, Azure, or Google Cloud
Build a REST API using Flask or FastAPI for integration with other applications
6. Multilingual Support
Extend the model to classify messages in multiple languages (e.g., Hindi and English)
Use language detection and translation techniques
7. Model Evaluation Improvements
Use cross-validation instead of a single train-test split
Track additional metrics such as ROC-AUC and confusion matrix
8. Continuous Learning
Implement a system to retrain the model periodically with new data
Allow user feedback to improve predictions over time
9. Enhanced User Interface
Improve the Streamlit UI with better visualization (graphs, prediction confidence)
Add message history and analytics dashboard
10. Spam Explainability
Show why a message is classified as spam using feature importance or explainability tools like LIME or SHAP
Increase transparency and trust in predictions

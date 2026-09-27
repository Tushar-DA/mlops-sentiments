# bow vs tfidf

# Import necessary libraries
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import os

import dagshub

# Initialize MLflow and DagsHub tracking
mlflow.set_tracking_uri('https://dagshub.com/Tushar-DA/mlops-sentiments.mlflow')
dagshub.init(repo_owner='Tushar-DA', repo_name='mlops-sentiments', mlflow=True)

# Load the data
df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv').drop(columns=['tweet_id'])

# Define text preprocessing functions
def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(df):
    """Normalize the text data."""
    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        return df
    except Exception as e:
        print(f'Error during text normalization: {e}')
        raise

# Normalize the text data
df = normalize_text(df)

# Filter dataset for target categories
x = df['sentiment'].isin(['happiness', 'sadness'])
df = df[x]

# Cast target column to explicit integer format
df['sentiment'] = df['sentiment'].replace({'sadness': 0, 'happiness': 1}).astype(int)

# Set the experiment name
mlflow.set_experiment("Exp 2: Bow vs TfIdf")

# Define feature extraction factory dict to generate fresh vectorizer instances
vectorizer_classes = {
    'BoW': CountVectorizer,
    'TF-IDF': TfidfVectorizer
}

# Define algorithm factory dict to generate fresh model instances
algorithm_classes = {
    'LogisticRegression': LogisticRegression,
    'MultinomialNB': MultinomialNB,
    'XGBoost': XGBClassifier,
    'RandomForest': RandomForestClassifier,
    'GradientBoosting': GradientBoostingClassifier
}

# Start the parent run
with mlflow.start_run(run_name="All Experiments") as parent_run:
    # Loop through algorithms and feature extraction methods (Child Runs)
    for algo_name, algo_class in algorithm_classes.items():
        for vec_name, vec_class in vectorizer_classes.items():
            with mlflow.start_run(run_name=f"{algo_name} with {vec_name}", nested=True) as child_run:
                
                # Instantiate fresh vectorizer for each iteration
                vectorizer = vec_class()
                X = vectorizer.fit_transform(df['content'])
                y = df['sentiment']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Log preprocessing parameters
                mlflow.log_param("vectorizer", vec_name)
                mlflow.log_param("algorithm", algo_name)
                mlflow.log_param("test_size", 0.2)
                
                # Instantiate fresh model for each iteration
                model = algo_class()
                model.fit(X_train, y_train)
                
                # Log model parameters matching your original logic
                if algo_name == 'LogisticRegression':
                    mlflow.log_param("C", model.C)
                elif algo_name == 'MultinomialNB':
                    mlflow.log_param("alpha", model.alpha)
                elif algo_name == 'XGBoost':
                    mlflow.log_param("n_estimators", model.n_estimators)
                    mlflow.log_param("learning_rate", model.learning_rate)
                elif algo_name == 'RandomForest':
                    mlflow.log_param("n_estimators", model.n_estimators)
                    mlflow.log_param("max_depth", model.max_depth)
                elif algo_name == 'GradientBoosting':
                    mlflow.log_param("n_estimators", model.n_estimators)
                    mlflow.log_param("learning_rate", model.learning_rate)
                    mlflow.log_param("max_depth", model.max_depth)
                
                # Model evaluation
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                # Log evaluation metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                
                # Log model
                mlflow.sklearn.log_model(model, "Multi model")
                
                # Save and log source artifact safely
                if '__file__' in globals():
                    mlflow.log_artifact(__file__)
                
                # Print the results for verification
                print(f"Algorithm: {algo_name}, Feature Engineering: {vec_name}")
                print(f"Accuracy: {accuracy}")
                print(f"Precision: {precision}")
                print(f"Recall: {recall}")
                print(f"F1 Score: {f1}")
                print("-" * 50)
# updated model evaluation

import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import mlflow.sklearn
import os
from dotenv import load_dotenv

load_dotenv()

dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set in .env or system environment.")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

repo_owner = 'Tushar-DA'
repo_name = 'mlops-sentiments'

mlflow.set_tracking_uri(f'https://dagshub.com/{repo_owner}/{repo_name}.mlflow')

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except Exception as e:
        logger.error('Error loading model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data: %s', e)
        raise

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return evaluation metrics."""
    try:
        # Convert string classes to binary integers if present
        if len(y_test) > 0 and isinstance(y_test[0], str):
            label_map = {'sadness': 0, 'happiness': 1}
            y_test = np.array([label_map.get(val, val) for val in y_test])

        y_pred = clf.predict(X_test)
        if len(y_pred) > 0 and isinstance(y_pred[0], str):
            label_map = {'sadness': 0, 'happiness': 1}
            y_pred = np.array([label_map.get(val, val) for val in y_pred])

        if hasattr(clf, "predict_proba"):
            y_pred_proba = clf.predict_proba(X_test)
            if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] > 1:
                y_pred_proba = y_pred_proba[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
        else:
            auc = 0.0

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error saving metrics: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save model run ID and path to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error saving model info: %s', e)
        raise

def main():
    mlflow.set_experiment("dvc-pipeline")
    with mlflow.start_run() as run:
        try:
            clf = load_model('./models/model.pkl')
            test_data = load_data('./data/processed/test_bow.csv')
            
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            metrics = evaluate_model(clf, X_test, y_test)
            
            save_metrics(metrics, 'reports/metrics.json')
            
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(clf, "model")
            
            # Save experiment info
            save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
            
            # Log artifacts
            mlflow.log_artifact('reports/metrics.json')
            mlflow.log_artifact('reports/experiment_info.json')

            if os.path.exists('model_evaluation_errors.log'):
                mlflow.log_artifact('model_evaluation_errors.log')
                
        except Exception as e:
            logger.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")
            raise e

if __name__ == '__main__':
    main()
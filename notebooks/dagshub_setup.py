import dagshub
import mlflow

mlflow.set_tracking_uri('https://dagshub.com/Tushar-DA/mlops-sentiments.mlflow')

dagshub.init(repo_owner='Tushar-DA', repo_name='mlops-sentiments', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
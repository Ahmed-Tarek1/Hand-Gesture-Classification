import mlflow
from mlflow.sklearn import log_model as mlflow_log_model

def set_experiment(experiment_name):
    mlflow.set_experiment(experiment_name)

def start_run(run_name):
    return mlflow.start_run(run_name=run_name)

def log_params(params: dict):
    mlflow.log_params(params)

def log_metrics(metrics: dict):
    mlflow.log_metrics(metrics)

def log_model(model, artifact_path="model"):
    mlflow_log_model(model, artifact_path)

def log_artifact(path):
    mlflow.log_artifact(path)

def register_model(model_uri, model_name):
    mlflow.register_model(model_uri, model_name)
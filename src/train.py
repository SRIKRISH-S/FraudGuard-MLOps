import os
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

from src.preprocessing import DataPreprocessor
from src.evaluate import calculate_metrics, plot_confusion_matrix, plot_roc_curve, plot_pr_curve

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ModelTraining")

# Set local MLflow tracking directory (using SQLite for registry metadata)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("FraudGuard_Model_Training")

class ModelTrainer:
    """
    Handles model training, comparison, MLflow tracking, and model registration.
    """
    def __init__(self, processed_dir: str = "data/processed", model_dir: str = "models"):
        self.processed_dir = processed_dir
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load datasets
        self.train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
        self.test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))
        
        # Initialize Preprocessor
        self.preprocessor = DataPreprocessor(apply_smote=True)
        
        # Feature Matrices
        self.X_train, self.y_train = self.preprocessor.fit_transform(self.train_df)
        self.X_test, self.y_test = self.preprocessor.transform(self.test_df)
        
        # Handle Imbalance on training set
        self.X_train_res, self.y_train_res = self.preprocessor.balance_training_data(
            self.X_train, self.y_train
        )
        
        # Save preprocessor
        self.preprocessor.save(os.path.join(self.model_dir, "preprocessor.pkl"))

    def train_logistic_regression(self) -> Tuple[LogisticRegression, Dict[str, float]]:
        """
        Trains Logistic Regression model.
        """
        logger.info("Training Logistic Regression Model...")
        model = LogisticRegression(max_iter=1000, random_state=42, solver="liblinear")
        
        with mlflow.start_run(run_name="Logistic_Regression"):
            model.fit(self.X_train_res, self.y_train_res)
            
            # Predict
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1]
            
            # Evaluate
            metrics = calculate_metrics(self.y_test, y_pred, y_prob)
            
            # Save and log plots
            cm_path = "logs/lr_cm.png"
            roc_path = "logs/lr_roc.png"
            pr_path = "logs/lr_pr.png"
            plot_confusion_matrix(self.y_test, y_pred, cm_path)
            plot_roc_curve(self.y_test, y_prob, roc_path)
            plot_pr_curve(self.y_test, y_prob, pr_path)
            
            # MLflow Log
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(cm_path)
            mlflow.log_artifact(roc_path)
            mlflow.log_artifact(pr_path)
            mlflow.sklearn.log_model(model, "model")
            
        logger.info("Logistic Regression model logged successfully.")
        return model, metrics

    def train_random_forest(self) -> Tuple[RandomForestClassifier, Dict[str, float]]:
        """
        Trains Random Forest Classifier.
        """
        logger.info("Training Random Forest Classifier...")
        # Restrict depth/estimators for faster demo training, still high quality
        model = RandomForestClassifier(
            n_estimators=50, 
            max_depth=12, 
            random_state=42, 
            n_jobs=-1
        )
        
        with mlflow.start_run(run_name="Random_Forest"):
            model.fit(self.X_train_res, self.y_train_res)
            
            # Predict
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1]
            
            # Evaluate
            metrics = calculate_metrics(self.y_test, y_pred, y_prob)
            
            # Save and log plots
            cm_path = "logs/rf_cm.png"
            roc_path = "logs/rf_roc.png"
            pr_path = "logs/rf_pr.png"
            plot_confusion_matrix(self.y_test, y_pred, cm_path)
            plot_roc_curve(self.y_test, y_prob, roc_path)
            plot_pr_curve(self.y_test, y_prob, pr_path)
            
            # MLflow Log
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(cm_path)
            mlflow.log_artifact(roc_path)
            mlflow.log_artifact(pr_path)
            mlflow.sklearn.log_model(model, "model")
            
        logger.info("Random Forest model logged successfully.")
        return model, metrics

    def train_xgboost(self) -> Tuple[XGBClassifier, Dict[str, float]]:
        """
        Trains XGBoost Classifier.
        """
        logger.info("Training XGBoost Classifier...")
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        )
        
        with mlflow.start_run(run_name="XGBoost"):
            model.fit(self.X_train_res, self.y_train_res)
            
            # Predict
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1]
            
            # Evaluate
            metrics = calculate_metrics(self.y_test, y_pred, y_prob)
            
            # Save and log plots
            cm_path = "logs/xgb_cm.png"
            roc_path = "logs/xgb_roc.png"
            pr_path = "logs/xgb_pr.png"
            plot_confusion_matrix(self.y_test, y_pred, cm_path)
            plot_roc_curve(self.y_test, y_prob, roc_path)
            plot_pr_curve(self.y_test, y_prob, pr_path)
            
            # MLflow Log
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(cm_path)
            mlflow.log_artifact(roc_path)
            mlflow.log_artifact(pr_path)
            # Log as XGBoost model type
            mlflow.xgboost.log_model(model, "model")
            
        logger.info("XGBoost model logged successfully.")
        return model, metrics

    def run_pipeline(self) -> None:
        """
        Runs the training pipeline, compares results, and registers the best model.
        """
        logger.info("Starting complete model training pipeline...")
        
        lr_model, lr_metrics = self.train_logistic_regression()
        rf_model, rf_metrics = self.train_random_forest()
        xgb_model, xgb_metrics = self.train_xgboost()
        
        # Compare F1 Score first, using ROC-AUC as secondary metric
        models_dict = {
            "LogisticRegression": (lr_model, lr_metrics),
            "RandomForest": (rf_model, rf_metrics),
            "XGBoost": (xgb_model, xgb_metrics)
        }
        
        best_name = None
        best_f1 = -1.0
        best_model = None
        best_metrics = {}
        
        for name, (model, metrics) in models_dict.items():
            f1 = metrics["f1_score"]
            logger.info(f"Model: {name} | F1: {f1:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                best_name = name
                best_model = model
                best_metrics = metrics
        
        logger.info(f"Winner model: {best_name} with F1-score of {best_f1:.4f}")
        
        # Save best model locally
        best_model_path = os.path.join(self.model_dir, "best_model.pkl")
        joblib.dump(best_model, best_model_path)
        logger.info(f"Saved best model locally at {best_model_path}")
        
        # Save model metadata
        metadata = {
            "model_type": best_name,
            "f1_score": best_metrics["f1_score"],
            "roc_auc": best_metrics["roc_auc"],
            "precision": best_metrics["precision"],
            "recall": best_metrics["recall"],
            "accuracy": best_metrics["accuracy"]
        }
        joblib.dump(metadata, os.path.join(self.model_dir, "model_metadata.pkl"))
        
        # Register model in MLflow registry
        with mlflow.start_run(run_name=f"Best_Model_Registration_{best_name}"):
            mlflow.log_params(metadata)
            mlflow.log_metrics(best_metrics)
            
            if best_name == "XGBoost":
                model_info = mlflow.xgboost.log_model(
                    best_model, 
                    "registered_model", 
                    registered_model_name="FraudGuard_Model"
                )
            else:
                model_info = mlflow.sklearn.log_model(
                    best_model, 
                    "registered_model", 
                    registered_model_name="FraudGuard_Model"
                )
            logger.info(f"Registered model in MLflow model registry: {model_info.model_uri}")
        
        logger.info("Pipeline run complete.")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run_pipeline()
import os
from joblib import dump
import joblib
from matplotlib import pyplot as plt
import mlflow
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
from config.constants import LOGISTIC_FEATURES, ModelName, ModelStage
from src.config.logger import Logger
from src.repository.stock_feature import StockFeatureRepository
from src.services.training.base import TrainMLModelBase
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature

class StockLogisticModel(TrainMLModelBase):
    def __init__(self):
        self.logger = Logger(StockLogisticModel.__name__)
        self.stock_feature_repository = StockFeatureRepository()

        load_dotenv()  # Load environment variables from .env file
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        mlflow_s3_endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL")
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.sklearn.autolog(disable=True)

        self.mlclient = mlflow.MlflowClient()

    def get_stock_features(self):
        stock_features = self.stock_feature_repository.get_training_features()

        return stock_features
    
    def prepare_data(self, LOGISTIC_FEATURES, test_size=0.2, random_state=42):
        self.logger.info("Preparing data for training...")
        
        stock_features = self.get_stock_features()
        stock_features_df = pd.DataFrame(stock_features)

        """Chuẩn bị dữ liệu train/test và scale"""
        X = stock_features_df[LOGISTIC_FEATURES].dropna()
        y = stock_features_df['label'].loc[X.index]

        # Chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        print(f"X_train shape: {X_test}, y_train shape: {y_test}")
        print("sklearn:", sklearn.__version__)
        print("numpy:", np.__version__)
        print("pandas:", pd.__version__)
        print("X shape:", X.shape)
        print("y distribution:", y.value_counts() if hasattr(y, "value_counts") else np.bincount(y))
        print("first rows:", X[:5])

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, scaler

    def train_logistic(self, X_train, y_train):
        self.logger.info("Training Logistic Regression model with GridSearchCV...")

        """Huấn luyện Logistic Regression với GridSearchCV"""
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs'],
            'max_iter': [1000]
        }
        model = LogisticRegression(class_weight='balanced')
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


    def evaluate_model(self, model, X_test, y_test):
        self.logger.info("Evaluating model on test set...")

        """Đánh giá mô hình trên tập test"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        metrics = {
            "accuracy": acc,
            "roc_auc": auc,
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1": report["1"]["f1-score"]
        }

        return metrics, cm


    def log_mlflow(self, 
        best_model, 
        best_params, 
        metrics, 
        scaler, 
        input_example,
        signature,
        model_name, 
        mlclient, 
        stage="Staging"
    ):
        self.logger.info("Logging model to MLflow...")
       
        """Log model, params, metrics lên MLflow"""
        with mlflow.start_run() as run:
            # Params
            mlflow.log_params(best_params)

            # Metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Artifacts: scaler + model
            joblib.dump(scaler, "scaler.pkl")
            mlflow.log_artifact("scaler.pkl", artifact_path="preprocessing")
            mlflow.sklearn.log_model(
                best_model, 
                artifact_path="model",
                input_example=input_example,
                signature=signature
            )

            # Đăng ký model
            result = mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name=model_name
            )

            # Chuyển stage
            mlclient.transition_model_version_stage(
                name=model_name,
                version=result.version,
                stage=stage
            )

            return run.info.run_id
        
    def run(self):
        # 1. Chuẩn bị dữ liệu
        X_train, X_test, y_train, y_test, scaler = self.prepare_data(LOGISTIC_FEATURES)

        # 2. Train model
        best_model, best_params, best_score = self.train_logistic(X_train, y_train)

        # 3. Đánh giá
        metrics, cm = self.evaluate_model(best_model, X_test, y_test)
        print("Confusion Matrix:\n", cm)
        print("Metrics:", metrics)

        # 4. Log MLflow
        input_example = X_train[:5]
        signature = infer_signature(X_train, best_model.predict(X_train))

        self.log_mlflow(
            best_model,
            best_params,
            metrics,
            scaler,
            input_example,
            signature,
            model_name=ModelName.STOCK_LOGISTIC_REGRESSION.value,
            mlclient=self.mlclient,
            stage=ModelStage.get_state()
        )

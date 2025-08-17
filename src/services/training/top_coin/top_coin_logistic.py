import os
from joblib import dump
import joblib
from matplotlib import pyplot as plt
import mlflow
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
from config.constants import LOGISTIC_FEATURES, TOP_COIN_LOGISTIC_FEATURES, ModelName, ModelStage
from src.repository.top_coin_feature import TopCoinFeatureRepository
from src.services.training.base import TrainMLModelBase
from dotenv import load_dotenv

class TopCoinLogisticModel(TrainMLModelBase):
    def __init__(self):
        self.top_coin_feature_repository = TopCoinFeatureRepository()

        load_dotenv()  # Load environment variables from .env file
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        mlflow_s3_endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL")
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.sklearn.autolog(disable=True)

        self.mlclient = mlflow.MlflowClient()

    def get_top_coin_features(self):
        top_coin_features = self.top_coin_feature_repository.get_training_features()

        return top_coin_features
    
    def run(self):
        top_coin_features = self.get_top_coin_features()
        top_coin_features_df = pd.DataFrame(top_coin_features)
        print(top_coin_features_df)
        # Chọn các feature để huấn luyện mô hình
        X = top_coin_features_df[TOP_COIN_LOGISTIC_FEATURES].dropna()  # Loại bỏ các giá trị NaN
        y = top_coin_features_df['label'].loc[X.index]  # Nhãn tương ứng với các feature

        # Kiểm tra tỷ lệ nhãn 1 và 0
        label_counts = y.value_counts()
        total_samples = len(y)
        label_ratio = label_counts / total_samples * 100  # Tính tỷ lệ phần trăm
        print("Tỷ lệ nhãn:")
        print(f"Nhãn 0: {label_counts.get(0, 0)} mẫu ({label_ratio.get(0, 0):.2f}%)")
        print(f"Nhãn 1: {label_counts.get(1, 0)} mẫu ({label_ratio.get(1, 0):.2f}%)")
        print(f"Tổng số mẫu: {total_samples}")

        # Chia tập train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Kiểm tra tỷ lệ nhãn trong tập train và test
        train_label_counts = y_train.value_counts()
        train_label_ratio = train_label_counts / len(y_train) * 100
        test_label_counts = y_test.value_counts()
        test_label_ratio = test_label_counts / len(y_test) * 100
        print("\nTỷ lệ nhãn trong tập train:")
        print(f"Nhãn 0: {train_label_counts.get(0, 0)} mẫu ({train_label_ratio.get(0, 0):.2f}%)")
        print(f"Nhãn 1: {train_label_counts.get(1, 0)} mẫu ({train_label_ratio.get(1, 0):.2f}%)")
        print(f"\nTỷ lệ nhãn trong tập test:")
        print(f"Nhãn 0: {test_label_counts.get(0, 0)} mẫu ({test_label_ratio.get(0, 0):.2f}%)")
        print(f"Nhãn 1: {test_label_counts.get(1, 0)} mẫu ({test_label_ratio.get(1, 0):.2f}%)")

        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Huấn luyện mô hình Logistic Regression với GridSearchCV
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs'],
            'max_iter': [1000]
        }
        model = LogisticRegression(class_weight='balanced')
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)

        # Lấy mô hình tốt nhất
        best_model = grid_search.best_estimator_
        print("\nTham số tốt nhất:", grid_search.best_params_)
        print("Điểm số tốt nhất (trên tập train):", grid_search.best_score_)

        # Đánh giá mô hình trên tập test
        test_score = best_model.score(X_test_scaled, y_test)
        print("Điểm số trên tập test:", test_score)
        # Dự đoán trên tập test
        y_pred = best_model.predict_proba(X_test_scaled)
        np.set_printoptions(suppress=True, precision=5)


        # Lưu scaler vào file
        # dump(scaler, f'''./stock_logistic_scaler.pkl''')

        # # Lưu mô hình Logistic Regression ra file
        # dump(best_model, f'''./stock_logistic_model.joblib''')


        # Step 1: Get predicted probability of class 1
        # y_pred_proba = y_pred[:, 1]

        # # Step 3: Merge with feature data (Optional, depending on need)
        # merged_df = X_test.reset_index(drop=True).copy()
        # merged_df['actual_label'] = y_test.reset_index(drop=True)
        # merged_df['predicted_proba'] = y_pred_proba

        # # Or merge into full original DataFrame
        # merged_full_df = stock_features_df.loc[y_test.index].copy()
        # merged_full_df['actual_label'] = y_test.values
        # merged_full_df['predicted_proba'] = y_pred_proba

        # # View result
        # print(merged_full_df[['date', 'symbol', 'actual_label', 'predicted_proba']])

        # # Step 1: Predict on training set
        # y_train_pred = best_model.predict_proba(X_train_scaled)
        # y_train_pred_proba = y_train_pred[:, 1]

        # # Step 2: Merge with full original DataFrame
        # merged_full_train_df = stock_features_df.loc[y_train.index].copy()
        # merged_full_train_df['actual_label'] = y_train.values
        # merged_full_train_df['predicted_proba'] = y_train_pred_proba

        # # Step 3: View result
        # print(merged_full_train_df[['date', 'symbol', 'actual_label', 'predicted_proba']])

        correlations = top_coin_features_df[TOP_COIN_LOGISTIC_FEATURES].apply(lambda x: x.corr(top_coin_features_df['label']))
        print(correlations.sort_values(ascending=False))


        # print(classification_report(y_test, best_model.predict(X_test_scaled)))
        # print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))


        # Log model to MLflow
        with mlflow.start_run() as run:
            # Log best parameters
            mlflow.log_params(grid_search.best_params_)

            # Log model metrics
            acc = best_model.score(X_test, y_test)
            mlflow.log_metric("accuracy", acc)

            # Log the best model
            joblib.dump(scaler, "scaler.pkl")
            mlflow.log_artifact("scaler.pkl", artifact_path="preprocessing")
            mlflow.sklearn.log_model(best_model, artifact_path="model")

            # Register the model
            result = mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name=ModelName.TOP_COIN_LOGISTIC_REGRESSION.value
            )

            # Transition the model version to "Staging"
            self.mlclient.transition_model_version_stage(
                name=ModelName.TOP_COIN_LOGISTIC_REGRESSION.value,
                version=result.version,
                stage=ModelStage.get_state()  # Use the ModelStage enum to get the current stage
            )
import os
from dotenv import load_dotenv
import joblib
# import mlflow
import pandas as pd
from config.constants import STOCK_LOGISTIC_FEATURES, ModelName, ModelStage
from src.config.logger import Logger
from src.repository.prediction_result import PredictionResultRepository
from src.repository.stock_feature import StockFeatureRepository
from src.services.prediction.base import PredictionBase


class StockLogisticPrediction(PredictionBase):
    def __init__(self, date):
        self.date = date
        self.logger = Logger(StockLogisticPrediction.__name__)

        self.stock_feature_repository = StockFeatureRepository()
        self.prediction_result_repository = PredictionResultRepository()

        load_dotenv()  # Load environment variables from .env file
        # mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        # mlflow_s3_endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL")
        # aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        # aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        
        # mlflow.set_tracking_uri(mlflow_tracking_uri)
        # mlflow.sklearn.autolog(disable=True)

        # self.mlclient = mlflow.MlflowClient()

    def get_stock_features(self):
        stock_features = self.stock_feature_repository.get_prediction_features(date=self.date)

        return stock_features

    def get_scaler(self):
        # Get latest model version in Staging
        # latest_versions = self.mlclient.get_latest_versions(
        #     name=ModelName.STOCK_LOGISTIC_REGRESSION.value, 
        #     stages=[ModelStage.get_state()]
        # )

        # latest_version = latest_versions[0]
        # run_id = latest_version.run_id

        # # Download scaler.pkl from that run
        # scaler_path = mlflow.artifacts.download_artifacts(
        #     artifact_uri=f"runs:/{run_id}/preprocessing/scaler.pkl"
        # )

        # Load scaler
        # scaler = joblib.load(scaler_path)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "..", "..", '..', "model", "StockLogisticRegression_scaler.pkl")
        scaler = joblib.load(model_path)  # Optional: save a local copy
        return scaler
    
    def get_model(self):
        # Load model from "Staging"
        # model_uri = f'''models:/{ModelName.STOCK_LOGISTIC_REGRESSION.value}/{ModelStage.get_state()}'''
        # model = mlflow.sklearn.load_model(model_uri)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "..", "..", '..', "model", "StockLogisticRegression_model.pkl")
        model = joblib.load(model_path)
        return model
    
    def run(self):
        self.logger.info(f"Start stock logistic prediction for date: {self.date}")
        
        stock_features = self.get_stock_features()
        if len(stock_features) == 0:
            self.logger.info(f"No stock features found for date: {self.date}")
            return
        
        stock_features_df = pd.DataFrame(stock_features)
        stock_features_df.fillna(-10e10, inplace=True)

        scaler_loaded = self.get_scaler()
        model = self.get_model()

        prediction_df = pd.DataFrame()

        for index, row in stock_features_df.iterrows():
            X = pd.DataFrame([row[STOCK_LOGISTIC_FEATURES]], columns=STOCK_LOGISTIC_FEATURES)
            X_scaler = scaler_loaded.transform(X)  # Scale the data
            
            # Predict on the test set
            y_pred = model.predict_proba(X_scaler)
            prediction_df = pd.concat([prediction_df, pd.DataFrame({'prediction': [y_pred[0][1]]})], axis=0, ignore_index=True)

        result_df = pd.concat([stock_features_df, prediction_df], axis=1)
        result_df['version'] = 'v1'

        result_df = result_df[['date', 'symbol', 'version', 'prediction', 'type']]
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]

        print(result_df)
        self.prediction_result_repository.delete_by_list_key(result_df.to_dict(orient='records'))
        self.prediction_result_repository.save_from_dataframe(result_df)

        self.logger.info(f"Finish stock logistic prediction for date: {self.date}")
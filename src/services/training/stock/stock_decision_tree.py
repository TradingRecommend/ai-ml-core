import os
import joblib
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature
from config.constants import STOCK_LOGISTIC_FEATURES, ModelName, ModelStage
from src.config.logger import Logger
from src.repository.stock_feature import StockFeatureRepository
from src.services.training.base import TrainMLModelBase
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

class StockDecisionTreeModel(TrainMLModelBase):
    def __init__(self):
        self.logger = Logger(StockDecisionTreeModel.__name__)
        self.stock_feature_repository = StockFeatureRepository()

        load_dotenv()
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.sklearn.autolog(disable=True)

        self.mlclient = mlflow.MlflowClient()

    def get_stock_features(self):
        return pd.DataFrame(self.stock_feature_repository.get_training_features())

    def prepare_data(self, FEATURES, test_size=0.2, random_state=42):
        self.logger.info("Preparing data for Decision Tree training...")
        df = self.get_stock_features()
        # basic checks
        if df.empty:
            raise ValueError("No training data available")

        # remove outliers based on provided features
        outliers = self.detect_outliers(df, FEATURES)
        if outliers:
            df = df.drop(index=outliers)

        X = df[FEATURES].dropna()
        y = df['label'].loc[X.index]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        # optional significance check (keeps parity with logistic flow)
        try:
            summary = self.check_feature_significance(X_train, y_train, FEATURES)
            self.logger.info(f"Feature significance:\n{summary}")
        except Exception:
            self.logger.info("Feature significance check failed or not applicable for decision tree.")

        return X_train, X_test, y_train, y_test

    def train_decision_tree(self, X_train, y_train):
        self.logger.info("Training Decision Tree with GridSearchCV...")

        param_grid = {
            "max_depth": [3, 5, 7, 9, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"]
        }
        model = DecisionTreeClassifier(class_weight="balanced", random_state=42)
        grid = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train, y_train)

        best = grid.best_estimator_
        return best, grid.best_params_, grid.best_score_

    def evaluate_model(self, model, X_test, y_test):
        self.logger.info("Evaluating Decision Tree model...")
        y_pred = model.predict(X_test)
        # some classifiers may not implement predict_proba for all cases; guard it
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = float("nan")

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        metrics = {
            "accuracy": acc,
            "roc_auc": auc,
            "precision": report.get("1", {}).get("precision", 0.0),
            "recall": report.get("1", {}).get("recall", 0.0),
            "f1": report.get("1", {}).get("f1-score", 0.0)
        }

        return metrics, cm

    def log_mlflow(self,
                   best_model,
                   best_params,
                   metrics,
                   input_example,
                   signature,
                   model_name,
                   mlclient,
                   stage="Staging"):
        self.logger.info("Logging Decision Tree model to MLflow...")
        with mlflow.start_run() as run:
            mlflow.log_params(best_params)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # log feature importances
            try:
                importances = getattr(best_model, "feature_importances_", None)
                if importances is not None:
                    fi = dict(zip(input_example.columns.tolist(), importances))
                    mlflow.log_dict(fi, "feature_importances.json")
            except Exception:
                pass

            # optional permutation importance (can be expensive)
            try:
                perm = permutation_importance(best_model, input_example, best_model.predict(input_example), n_repeats=5, random_state=42, n_jobs=-1)
                perm_df = dict(zip(input_example.columns.tolist(), perm.importances_mean.tolist()))
                mlflow.log_dict(perm_df, "permutation_importance.json")
            except Exception:
                pass

            mlflow.sklearn.log_model(
                best_model,
                artifact_path="model",
                input_example=input_example.head(5),
                signature=signature
            )

            result = mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name=model_name
            )

            mlclient.transition_model_version_stage(
                name=model_name,
                version=result.version,
                stage=stage
            )

            return run.info.run_id

    def detect_outliers(self, df, features, threshold=1.5):
        self.logger.info("Detecting outliers using IQR...")
        outlier_indices = set()
        for col in features:
            if col not in df.columns:
                continue
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            outliers = df[(df[col] < lower) | (df[col] > upper)].index
            outlier_indices.update(outliers)
        return list(outlier_indices)

    def calculate_vif(self, X, feature_names):
        self.logger.info("Calculating VIF to detect multicollinearity...")
        X_np = np.asarray(X)
        vif_data = pd.DataFrame()
        vif_data["feature"] = feature_names
        vif_data["VIF"] = [variance_inflation_factor(X_np, i) for i in range(X_np.shape[1])]
        return vif_data

    def check_feature_significance(self, X, y, feature_names):
        self.logger.info("Checking feature significance (statsmodels) ...")
        X_const = sm.add_constant(X)
        logit = sm.Logit(y, X_const)
        res = logit.fit(disp=0)
        summary_df = pd.DataFrame({
            "feature": ["const"] + feature_names,
            "coef": res.params.values,
            "p_value": res.pvalues.values
        })
        return summary_df

    def run(self):
        X_train, X_test, y_train, y_test = self.prepare_data(STOCK_LOGISTIC_FEATURES)

        best_model, best_params, best_score = self.train_decision_tree(X_train, y_train)

        metrics, cm = self.evaluate_model(best_model, X_test, y_test)
        self.logger.info(f"Confusion Matrix:\n{cm}")
        self.logger.info(f"Metrics: {metrics}")

        input_example = X_train.head(5)
        signature = infer_signature(input_example, best_model.predict(input_example))

        self.log_mlflow(
            best_model,
            best_params,
            metrics,
            input_example,
            signature,
            model_name=ModelName.STOCK_DECISION_TREE.value,
            mlclient=self.mlclient,
            stage=ModelStage.get_state()
        )

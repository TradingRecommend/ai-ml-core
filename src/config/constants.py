from enum import Enum
import os

from dotenv import load_dotenv


MODEL = {
    'LOGISTIC': 'LOGISTIC',
    'XGBOOST': 'XGBOOST',
    'LIGHTGBM': 'LIGHTGBM',
    'RANDOM_FOREST': 'RANDOM_FOREST',
}

TOP_COIN_LOGISTIC_FEATURES = ['price_change_ratio', 'volume_mean_7_days_ratio', 'volume_mean_20_days_ratio', 'volume_mean_50_days_ratio', 'MA7', 'MA20', 'MA50', 'RSI', 'MACD',
            'MACD_signal', 'MACD_hist', 'Stoch_K', 'Stoch_D', 'ADX', 'MFI']

LOGISTIC_FEATURES = ['price_change_ratio', 'volume_mean_7_days_ratio', 'volume_mean_20_days_ratio', 'MA7', 'MA50', 'MACD',
            'MACD_signal', 'ADX', 'MFI']

# LOGISTIC_FEATURES = ['volume_mean_7_days_ratio', 'volume_mean_20_days_ratio', 'MA7', 'MA20', 'MA50', 'MACD', 'ADX', 'MFI']

class TradeType(Enum):
    STOCK = "1"
    TOP_COIN = "2"
    PENNY_COIN = "3"
    FUTURE = "4"

    @classmethod
    def choices(cls):
        return [(cls.STOCK.value, "Stock"), (cls.COIN.value, "Coin"), (cls.FUTURE.value, "Future")]

class ModelType(Enum):
    LOGISTIC = "LOGISTIC"
    XGBOOST = "XGBOOST"
    LIGHTGBM = "LIGHTGBM"
    RANDOM_FOREST = "RANDOM_FOREST"


class ModelStage(Enum):
    STAGING = "Staging"
    PRODUCTION = "Production"

    @staticmethod
    def get_state():
        load_dotenv()
        mode = os.getenv("MODE")

        if mode == "prod":
            return ModelStage.PRODUCTION.value
        elif mode == "staging":
            return ModelStage.STAGING.value
        
class ModelName(Enum):
    STOCK_LOGISTIC_REGRESSION = "StockLogisticRegression"
    TOP_COIN_LOGISTIC_REGRESSION = "TopCoinLogisticRegression"
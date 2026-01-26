from enum import Enum
import os

from dotenv import load_dotenv


MODEL = {
    'LOGISTIC': 'LOGISTIC',
    'XGBOOST': 'XGBOOST',
    'LIGHTGBM': 'LIGHTGBM',
    'RANDOM_FOREST': 'RANDOM_FOREST',
}

TOP_COIN_LOGISTIC_FEATURES = ['price_change_ratio', 'volume_mean_7_days_ratio', 'volume_mean_20_days_ratio', 'MA26_trend', 'MA50_trend', 'MA9_MA26_ratio', 'MA9_MA50_ratio', 'MA26_MA50_ratio', 'MACD',
            'MACD_signal', 'ADX']

STOCK_LOGISTIC_FEATURES = ['price_change_ratio', 'volume_mean_7_days_ratio', 'volume_mean_20_days_ratio', 'MA26_trend', 'MA50_trend', 'MA9_MA26_ratio', 'MA9_MA50_ratio', 'MA26_MA50_ratio', 'MACD',
            'MACD_signal', 'ADX']

PENNY_COIN_LOGISTIC_FEATURES = ['price_change_ratio', 'volume_mean_7_days_ratio', 'volume_mean_20_days_ratio', 'volume_mean_50_days_ratio', 'MA7', 'MA20', 'MA50', 'RSI', 'MACD',
            'MACD_signal', 'MACD_hist', 'Stoch_K', 'Stoch_D', 'ADX', 'MFI']

LOGISTIC_FEATURES = ['price_change_ratio', 'volume_mean_7_days_ratio', 'volume_mean_20_days_ratio', 'volume_mean_50_days_ratio', 'MA7', 'MA20', 'MA50', 'RSI', 'MACD',
            'MACD_signal', 'MACD_hist', 'Stoch_K', 'Stoch_D', 'ADX', 'MFI']

class TradeType(Enum):
    STOCK = "1"
    TOP_COIN = "2"
    PENNY_COIN = "3"
    FUTURE = "4"

    @classmethod
    def choices(cls):
        return [(cls.STOCK.value, "Stock"), (cls.TOP_COIN.value, "Top Coin"), (cls.FUTURE.value, "Future"), (cls.PENNY_COIN.value, "Penny Coin")]

class ModelType(Enum):
    LOGISTIC = "LOGISTIC"
    DECISION_TREE = "DECISION_TREE"
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
    STOCK_DECISION_TREE = "StockDecisionTree"
    TOP_COIN_LOGISTIC_REGRESSION = "TopCoinLogisticRegression"
    PENNY_COIN_LOGISTIC_REGRESSION = "PennyCoinLogisticRegression"
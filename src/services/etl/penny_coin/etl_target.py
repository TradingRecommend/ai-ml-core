from typing import List

import requests

from src.config.constants import TradeType
from src.config.logger import Logger
from src.entities.target import TargetEntity
from src.repository.target import TargetRepository
from src.repository.top_coin_feature import TopCoinFeatureRepository
from src.services.etl.base import ETLBase
from src.services.etl.top_coin.build_feature import BuildTopCoinFeature

class ETLPennyCoinTarget(ETLBase):
    def __init__(self):
        self.logger = Logger(ETLPennyCoinTarget.__name__)
        
        self.target_repository = TargetRepository()

    def get_penny_coin_targets(self) -> List[TargetEntity]:
        url = "https://api.binance.com/api/v3/exchangeInfo"
        response = requests.get(url)
        data = response.json()
        symbols = []

        for symbol_info in data.get('symbols'):
            if not (symbol_info.get('status') == 'TRADING' and symbol_info.get('quoteAsset') == 'USDT' and symbol_info.get('isSpotTradingAllowed')):
                continue

            price_filter = [filter for filter in symbol_info.get('filters') if filter.get('filterType') == 'PRICE_FILTER'][0]
            if not (float(price_filter.get('tickSize', '0')) <= 0.0001):
                continue

            price_filter = [filter for filter in symbol_info.get(
                'filters') if filter.get('filterType') == 'LOT_SIZE'][0]
            if float(price_filter.get('minQty', '0')) == 1 and float(price_filter.get('stepSize', '0')) == 1 and float(price_filter.get('maxQty', '0')) == 10000000:
                continue

            symbols.append(symbol_info.get('symbol')[:-4])  # Remove 'USDT' suffix

        penny_coin_targets = [TargetEntity(symbol=symbol, type=TradeType.PENNY_COIN.value, status=1) for symbol in symbols]
        return penny_coin_targets

    def extract(self):
        self.logger.info(f"Start extract penny coin targets")

        penny_coin_targets = self.get_penny_coin_targets()

        self.logger.info(f"Finish extract penny coin targets")

        return penny_coin_targets
    
    def transform(self, penny_coin_targets: List[TargetEntity]):
     
        return penny_coin_targets
    
    def load(self, penny_coin_targets):
        self.logger.info(f"Start load penny coin targets")

        self.target_repository.delete_by_type_value(type=TradeType.PENNY_COIN.value)
        self.target_repository.save_multiple(penny_coin_targets)

        self.logger.info(f"Finish load penny coin targets")
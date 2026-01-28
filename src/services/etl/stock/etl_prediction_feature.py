from time import sleep
from typing import List

from src.config.constants import TradeType
from src.config.logger import Logger
from src.entities.target import TargetEntity
from src.repository.stock_feature import StockFeatureRepository
from src.repository.target import TargetRepository
from src.services.etl.base import ETLBase
from src.services.etl.stock.build_feature import BuildStockFeature

class ETLPredictionStockFeature(ETLBase):
    def __init__(self, date: str):
        self.date = date
        self.logger = Logger(ETLPredictionStockFeature.__name__)
        
        self.target_repository = TargetRepository()
        self.feature_repository = StockFeatureRepository()

    def get_stock_targets(self) -> List[TargetEntity]:
        stock_targets = self.target_repository.get_by_type(type=TradeType.STOCK.value)  # Assuming "1" is the type for stock labels
        if not stock_targets:
            raise ValueError("No stock targets found in the database.")

        # Set each target's date to the specified date
        for target in stock_targets:
            target.date = self.date
        return stock_targets

    def extract(self):
        self.logger.info(f"Start extract stock targets for date: {self.date}")

        stock_targets = self.get_stock_targets()

        self.logger.info(f"Finish extract {len(stock_targets)} stock targets for date: {self.date}")

        return stock_targets
    
    def transform(self, stock_targets: List[TargetEntity]):
        self.logger.info(f"Start transform stock targets for date: {self.date}")

        feature_df = BuildStockFeature.build_stock_features([stock_target.__dict__ for stock_target in stock_targets])

        self.logger.info(f"Finish transform stock targets for date: {self.date}")

        return feature_df
    
    def load(self, feature_df):
        self.logger.info(f"Start load stock features for date: {self.date}")

        retry_max = 5
        is_done = False
        retry_cnt = 0
        while not is_done:
            try:
                self.feature_repository.delete_by_date(self.date)
                self.feature_repository.save_from_dataframe(feature_df)
                is_done = True
            except Exception as e:  
                self.logger.error(f"Error loading stock features for date: {self.date}. Error: {e}")
                retry_cnt += 1
                if retry_cnt >= retry_max:
                    self.logger.error(f"Max retries reached for loading stock features for date: {self.date}. Skipping...")
                    is_done = True
                else:
                    self.logger.info("Retrying in 60 seconds...")
                    sleep(60)

        self.logger.info(f"Finish load stock features for date: {self.date}")
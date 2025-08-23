from typing import List

from src.config.constants import TradeType
from src.config.logger import Logger
from src.entities.target import TargetEntity
from src.repository.target import TargetRepository
from src.repository.top_coin_feature import TopCoinFeatureRepository
from src.services.etl.base import ETLBase
from src.services.etl.top_coin.build_feature import BuildTopCoinFeature

class ETLPredictionTopCoinFeature(ETLBase):
    def __init__(self, date: str):
        self.date = date
        self.logger = Logger(ETLPredictionTopCoinFeature.__name__)
        
        self.target_repository = TargetRepository()
        self.feature_repository = TopCoinFeatureRepository()

    def get_top_coin_targets(self) -> List[TargetEntity]:
        top_coin_targets = self.target_repository.get_by_type(type=TradeType.TOP_COIN.value)  # Assuming "1" is the type for stock labels
        if not top_coin_targets:
            raise ValueError("No top coin targets found in the database.")

        # Set each target's date to the specified date
        for target in top_coin_targets:
            target.date = self.date
        return top_coin_targets

    def extract(self):
        self.logger.info(f"Start extract top coin targets for date: {self.date}")

        top_coin_targets = self.get_top_coin_targets()

        self.logger.info(f"Finish extract {len(top_coin_targets)} top coin targets for date: {self.date}")

        return top_coin_targets
    
    def transform(self, top_coin_targets: List[TargetEntity]):
        self.logger.info(f"Start transform top coin targets for date: {self.date}")

        feature_df = BuildTopCoinFeature.build_top_coin_features([top_coin_target.__dict__ for top_coin_target in top_coin_targets])

        self.logger.info(f"Finish transform top coin targets for date: {self.date}")

        return feature_df
    
    def load(self, feature_df):
        self.logger.info(f"Start load top coin features for date: {self.date}")

        self.feature_repository.delete_by_list_symbol_and_date(feature_df.to_dict(orient='records'))
        self.feature_repository.save_from_dataframe(feature_df)

        self.logger.info(f"Finish load top coin features for date: {self.date}")
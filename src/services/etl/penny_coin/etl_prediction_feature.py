from typing import List

from src.config.constants import TradeType
from src.config.logger import Logger
from src.entities.target import TargetEntity
from src.repository.penny_coin_feature import PennyCoinFeatureRepository
from src.repository.target import TargetRepository
from src.services.etl.base import ETLBase
from src.services.etl.penny_coin.build_feature import BuildPennyCoinFeature

class ETLPredictionPennyCoinFeature(ETLBase):
    def __init__(self, date: str):
        self.date = date
        self.logger = Logger(ETLPredictionPennyCoinFeature.__name__)
        
        self.target_repository = TargetRepository()
        self.feature_repository = PennyCoinFeatureRepository()

    def get_penny_coin_targets(self) -> List[TargetEntity]:
        targets = self.target_repository.get_by_type(type=TradeType.PENNY_COIN.value)  # Assuming "1" is the type for stock labels
        if not targets:
            raise ValueError("No penny coin targets found in the database.")

        # Set each target's date to the specified date
        for target in targets:
            target.date = self.date
        return targets

    def extract(self):
        self.logger.info(f"Start extract penny coin targets for date: {self.date}")

        targets = self.get_penny_coin_targets()

        self.logger.info(f"Finish extract {len(targets)} penny coin targets for date: {self.date}")

        return targets
    
    def transform(self, targets: List[TargetEntity]):
        self.logger.info(f"Start transform penny coin targets for date: {self.date}")

        feature_df = BuildPennyCoinFeature.build_penny_coin_features([target.__dict__ for target in targets])

        self.logger.info(f"Finish transform penny coin targets for date: {self.date}")

        return feature_df
    
    def load(self, feature_df):
        self.logger.info(f"Start load penny coin features for date: {self.date}")

        self.feature_repository.delete_by_list_symbol_and_date(feature_df.to_dict(orient='records'))
        self.feature_repository.save_from_dataframe(feature_df)

        self.logger.info(f"Finish load penny coin features for date: {self.date}")
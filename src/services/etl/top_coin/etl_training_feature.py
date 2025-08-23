from typing import List

from src.config.logger import Logger
from src.entities.label import LabelEntity
from src.repository.label import LabelRepository
from src.repository.top_coin_feature import TopCoinFeatureRepository
from src.services.etl.base import ETLBase
from src.services.etl.top_coin.build_feature import BuildTopCoinFeature

class ETLTrainingTopCoinFeature(ETLBase):
    def __init__(self):
        self.logger = Logger(ETLTrainingTopCoinFeature.__name__)

        self.label_repository = LabelRepository()
        self.feature_repository = TopCoinFeatureRepository()

    def get_top_coin_labels(self) -> List[LabelEntity]:
        top_coin_labels = self.label_repository.get_labels_not_in_top_coin_features()  # Assuming "1" is the type for stock labels

        return top_coin_labels

    def extract(self):
        self.logger.info(f"Start extract top coin trainin features")

        top_coin_labels = self.get_top_coin_labels()

        self.logger.info(f"Finish extract top coin trainin features")

        return top_coin_labels
    
    def transform(self, top_coin_labels: List[LabelEntity]):
        self.logger.info(f"Start transform top coin trainin features")
        
        feature_df = BuildTopCoinFeature.build_top_coin_features(LabelEntity.to_dict(top_coin_labels))

        self.logger.info(f"Finish transform top coin trainin features")

        return feature_df
    
    def load(self, feature_df):
        self.logger.info(f"Start load top coin trainin features")
       
        self.feature_repository.delete_by_list_symbol_and_date(feature_df.to_dict(orient='records'))
        self.feature_repository.save_from_dataframe(feature_df)

        self.logger.info(f"Finish load top coin trainin features")
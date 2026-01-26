from typing import List

from src.config.logger import Logger
from src.entities.label import LabelEntity
from src.repository.label import LabelRepository
from src.repository.penny_coin_feature import PennyCoinFeatureRepository
from src.services.etl.base import ETLBase
from src.services.etl.penny_coin.build_feature import BuildPennyCoinFeature

class ETLTrainingPennyCoinFeature(ETLBase):
    def __init__(self):
        self.logger = Logger(ETLTrainingPennyCoinFeature.__name__)

        self.label_repository = LabelRepository()
        self.feature_repository = PennyCoinFeatureRepository()

    def get_penny_coin_labels(self) -> List[LabelEntity]:
        print(1)
        labels = self.label_repository.get_labels_not_in_penny_coin_features()  # Assuming "1" is the type for stock labels
        print(labels)
        return labels

    def extract(self):
        self.logger.info(f"Start extract penny coin trainin features")

        labels = self.get_penny_coin_labels()

        self.logger.info(f"Finish extract penny coin trainin features")

        return labels
    
    def transform(self, labels: List[LabelEntity]):
        self.logger.info(f"Start transform penny coin trainin features")
        
        feature_df = BuildPennyCoinFeature.build_penny_coin_features(LabelEntity.to_dict(labels))

        self.logger.info(f"Finish transform penny coin trainin features")

        return feature_df
    
    def load(self, feature_df):
        self.logger.info(f"Start load penny coin trainin features")
       
        self.feature_repository.delete_by_list_symbol_and_date(feature_df.to_dict(orient='records'))
        self.feature_repository.save_from_dataframe(feature_df)

        self.logger.info(f"Finish load penny coin trainin features")
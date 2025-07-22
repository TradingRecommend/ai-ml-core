from typing import List

from src.config.constants import TradeType
from src.entities.label import LabelEntity
from src.repository.label import LabelRepository
from src.repository.stock_feature import StockFeatureRepository
from src.services.etl.base import ETLBase
from src.services.etl.stock.build_feature import BuildStockFeature

class ETLTrainingStockFeature(ETLBase):
    def __init__(self):
        self.label_repository = LabelRepository()
        self.feature_repository = StockFeatureRepository()

    def get_stock_labels(self) -> List[LabelEntity]:
        stock_labels = self.label_repository.get_labels_not_in_stock_features(type=TradeType.STOCK.value)  # Assuming "1" is the type for stock labels

        return stock_labels

    def extract(self):
        stock_labels = self.get_stock_labels()

        return stock_labels
    
    def transform(self, stock_labels: List[LabelEntity]):
        feature_df = BuildStockFeature.build_stock_features(LabelEntity.to_dict(stock_labels))

        return feature_df
    
    def load(self, feature_df):
        self.feature_repository.delete_by_list_symbol_and_date(feature_df.to_dict(orient='records'))
        self.feature_repository.save_from_dataframe(feature_df)

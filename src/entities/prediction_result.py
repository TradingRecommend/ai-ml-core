import pandas as pd
from sqlalchemy import Column, Double, String, CHAR, Enum as SQLEnum
from src.config.constants import TradeType
from src.database.db import Base

class PredictionResultEntity(Base):
    __tablename__ = "prediction_result"

    symbol = Column(String, primary_key=True)
    date = Column(CHAR(8), primary_key=True)
    version = Column(String, primary_key=True)
    type = Column(String, primary_key=True, comment="1: stock, 2: top-coin, 3: penny-coin, 4: future")
    prediction = Column(Double, nullable=False)


    @staticmethod
    def to_dataframe(entities: list) -> pd.DataFrame:
        """
        Convert a list of StockLabelEntity objects to a pandas DataFrame.
        """
        data = [
            {
                "symbol": e.symbol,
                "date": e.date,
                "version": e.version,
                "type": e.type,
                "prediction": e.prediction,
            }
            for e in entities
        ]
        return pd.DataFrame(data)
    
    @staticmethod
    def to_dict(entities: list) -> list:
        """
        Convert a list of TargetEntity objects to a list of dictionaries.
        """
        return [
            {
                "symbol": e.symbol,
                "date": e.date,
                "version": e.version,
                "type": e.type,
                "prediction": e.prediction,
            }
            for e in entities
        ]
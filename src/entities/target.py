import pandas as pd
from sqlalchemy import Column, Double, String, CHAR, Enum as SQLEnum, Integer
from src.config.constants import TradeType
from src.database.db import Base

class TargetEntity(Base):
    __tablename__ = "targets"

    symbol = Column(CHAR(3), primary_key=True)
    type = Column(String, primary_key=True, comment="1: stock, 2: coin, 3: future")
    status = Column(Integer, nullable=False)


    @staticmethod
    def to_dataframe(entities: list) -> pd.DataFrame:
        """
        Convert a list of StockLabelEntity objects to a pandas DataFrame.
        """
        data = [
            {
                "symbol": e.symbol,
                "type": e.type,
                "status": e.status,
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
                "type": e.type,
                "status": e.status,
            }
            for e in entities
        ]
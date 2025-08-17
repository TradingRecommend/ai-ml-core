from enum import Enum
import pandas as pd
from sqlalchemy import Column, Integer, String, Date, CHAR, Enum as SQLEnum
from src.config.constants import TradeType
from src.database.db import Base


class LabelEntity(Base):
    __tablename__ = "labels"

    symbol = Column(String, primary_key=True)
    date = Column(CHAR(8), primary_key=True)
    label = Column(Integer, nullable=False)
    type = Column(String, primary_key=True, comment="1: stock, 2: top-coin, 3: penny-coin, 4: future")


    @staticmethod
    def to_dataframe(entities: list) -> pd.DataFrame:
        """
        Convert a list of StockLabelEntity objects to a pandas DataFrame.
        """
        data = [
            {
                "symbol": e.symbol,
                "date": e.date,
                "label": e.label,
                "type": e.type,
            }
            for e in entities
        ]
        return pd.DataFrame(data)
    
    @staticmethod
    def to_dict(entities: list) -> list:
        """
        Convert a list of LabelEntity objects to a list of dictionaries.
        """
        return [
            {
                "symbol": e.symbol,
                "date": e.date,
                "label": e.label,
                "type": e.type,
            }
            for e in entities
        ]
    
    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> list:
        """
        Convert a pandas DataFrame to a list of LabelEntity objects.
        """
        return [
            LabelEntity(
                symbol=row['symbol'],
                date=row['date'],
                label=row['label'],
                type=row['type']
            )
            for _, row in df.iterrows()
        ]
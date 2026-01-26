from sqlalchemy import CHAR, String, create_engine, Column, Float, Integer
from sqlalchemy.orm import sessionmaker
from typing import List, Optional

from src.database.db import Base



class StockFeatureEntity(Base):
    __tablename__ = 'stock_feature'
    date = Column(CHAR(8),  primary_key=True)  # Assuming date is stored as an integer (e.g., YYYYMMDD)
    symbol = Column(String,  primary_key=True)
    volume = Column(Float)
    open = Column(Float)
    close = Column(Float)
    price_change_ratio = Column(Float)
    volume_mean_7_days_ratio = Column(Float)
    volume_mean_20_days_ratio = Column(Float)
    volume_mean_50_days_ratio = Column(Float)
    MA7 = Column(Float)
    MA9 = Column(Float)
    MA14 = Column(Float)
    MA18 = Column(Float)
    MA20 = Column(Float)
    MA26 = Column(Float)
    MA50 = Column(Float)
    MA52 = Column(Float)
    RSI = Column(Float)
    MACD = Column(Float)
    MACD_signal = Column(Float)
    MACD_hist = Column(Float)
    Stoch_K = Column(Float)
    Stoch_D = Column(Float)
    ADX = Column(Float)
    MFI = Column(Float)
    MA9_MA26_ratio = Column(Float)
    MA9_MA50_ratio = Column(Float)
    MA26_MA50_ratio = Column(Float)
    MA9_trend = Column(Float)
    MA26_trend = Column(Float)
    MA50_trend = Column(Float)
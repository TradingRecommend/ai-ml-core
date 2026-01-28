from typing import List, Optional

import pandas as pd
from sqlalchemy import text
from src.entities.stock_feature import StockFeatureEntity
from src.database.db import DatabaseManager

class StockFeatureRepository:
    def __init__(self):
        """Initialize repository with a DatabaseManager instance."""
        self.db_manager = DatabaseManager()
        StockFeatureEntity.metadata.create_all(self.db_manager.get_engine())
        
    def save_multiple(self, features: List[StockFeatureEntity]) -> None:
        """Save multiple feature entities to the database."""
        session = self.db_manager.Session()
        try:
            session.bulk_save_objects(features)
            session.commit()
        finally:
            session.close()

    def save(self, feature: StockFeatureEntity) -> None:
        """Save a single feature entity to the database."""
        session = self.db_manager.Session()
        try:
            session.add(feature)
            session.commit()
        finally:
            session.close()

    def delete_by_list_symbol_and_date(self, datas: List[object]) -> None:
        """
            Delete stock features that have (date, symbol) matching any labels in the list.
        """
        session = self.db_manager.Session()
        try:
            for data in datas:
                session.query(StockFeatureEntity).filter(
                    StockFeatureEntity.symbol == data.get('symbol'),
                    StockFeatureEntity.date == data.get('date')
                ).delete(synchronize_session=False)
            session.commit()
        finally:
            session.close()

    def delete_by_date(self, date: str) -> None:
        """
            Delete stock features that have (date, symbol) matching any labels in the list.
        """
        session = self.db_manager.Session()
        try:
            session.query(StockFeatureEntity).filter(
                StockFeatureEntity.date == date
            ).delete(synchronize_session=False)
            session.commit()
        finally:
            session.close()

    def get_all(self) -> List[StockFeatureEntity]:
        """Retrieve all feature entities."""
        session = self.db_manager.Session()
        try:
            return session.query(StockFeatureEntity).all()
        finally:
            session.close()

    # save data from pandas dataframe
    def save_from_dataframe(self, df: pd.DataFrame) -> None:
        engine = self.db_manager.get_engine()
        df.to_sql(
            name=StockFeatureEntity.__tablename__,  # Replace with your actual table name if different
            con=engine,
            if_exists='append',
            index=False,
        )

    # Get training features
    def get_training_features(self) -> list[object]:
        session = self.db_manager.Session()
        try:
            sql = """
                SELECT a.label, b.* FROM label a, stock_feature b
                where a.date = b.date
                    and a.symbol = b.symbol
                    and a.type = '1'
                order by b.date, b.symbol
            """
            rows = session.execute(text(sql)).mappings().all()
            return rows
        finally:
            session.close()

    # Get prediction features as pandas DataFrames
    def get_prediction_features(self, date: str) -> list[object]:
        session = self.db_manager.Session()
        try:
            sql = """
                SELECT a.type, b.* FROM target a, stock_feature b
                where b.date = :date
                    and a.symbol = b.symbol
                    and a.type = '1'
                    and a.status = '1'
            """
            rows = session.execute(text(sql), {'date': date}).mappings().all()
            return rows
        finally:
            session.close()
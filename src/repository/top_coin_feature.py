from typing import List, Optional

import pandas as pd
from sqlalchemy import text
from src.database.db import DatabaseManager
from src.entities.top_coin_feature import TopCoinFeatureEntity

class TopCoinFeatureRepository:
    def __init__(self):
        """Initialize repository with a DatabaseManager instance."""
        self.db_manager = DatabaseManager()
        TopCoinFeatureEntity.metadata.create_all(self.db_manager.get_engine())
        
    def save_multiple(self, features: List[TopCoinFeatureEntity]) -> None:
        """Save multiple feature entities to the database."""
        session = self.db_manager.Session()
        try:
            session.bulk_save_objects(features)
            session.commit()
        finally:
            session.close()

    def save(self, feature: TopCoinFeatureEntity) -> None:
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
                session.query(TopCoinFeatureEntity).filter(
                    TopCoinFeatureEntity.symbol == data.get('symbol'),
                    TopCoinFeatureEntity.date == data.get('date')
                ).delete(synchronize_session=False)
            session.commit()
        finally:
            session.close()

    def get_all(self) -> List[TopCoinFeatureEntity]:
        """Retrieve all feature entities."""
        session = self.db_manager.Session()
        try:
            return session.query(TopCoinFeatureEntity).all()
        finally:
            session.close()

    # save data from pandas dataframe
    def save_from_dataframe(self, df: pd.DataFrame) -> None:
        engine = self.db_manager.get_engine()
        df.to_sql(
            name=TopCoinFeatureEntity.__tablename__,  # Replace with your actual table name if different
            con=engine,
            if_exists='append',
            index=False,
        )

    # Get training features
    def get_training_features(self) -> list[object]:
        session = self.db_manager.Session()
        try:
            sql = """
                SELECT a.label, b.* FROM labels a, top_coin_features b
                where a.date = b.date
                    and a.symbol = b.symbol
                    and a.type = '2'
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
                SELECT a.type, b.* FROM targets a, top_coin_features b
                where b.date = :date
                    and a.symbol = b.symbol
                    and a.type = '1'
                    and a.status = '1'
            """
            rows = session.execute(text(sql), {'date': date}).mappings().all()
            return rows
        finally:
            session.close()
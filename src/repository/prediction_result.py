from typing import List, Optional
import pandas as pd
from sqlalchemy.orm import Session
from src.database.db import DatabaseManager
from src.entities.prediction_result import PredictionResultEntity

class PredictionResultRepository:
    def __init__(self):
        self.db_manager = DatabaseManager()
        PredictionResultEntity.metadata.create_all(self.db_manager.get_engine())

    def save(self, label: PredictionResultEntity) -> None:
        session = self.db_manager.Session()
        try:
            session.add(label)
            session.commit()
        finally:
            session.close()

    def save_multiple(self, targets: List[PredictionResultEntity]) -> None:
        session = self.db_manager.Session()
        try:
            session.bulk_save_objects(targets)
            session.commit()
        finally:
            session.close()

    # save data from pandas dataframe
    def save_from_dataframe(self, df: pd.DataFrame) -> None:
        engine = self.db_manager.get_engine()
        df.to_sql(
            name=PredictionResultEntity.__tablename__,  # Replace with your actual table name if different
            con=engine,
            if_exists='append',
            index=False,
        )

    def delete_by_list_key(self, datas: List[object]) -> None:
        """
            Delete stock features that have (date, symbol) matching any labels in the list.
        """
        session = self.db_manager.Session()
        try:
            for data in datas:
                session.query(PredictionResultEntity).filter(
                    PredictionResultEntity.symbol == data.get('symbol'),
                    PredictionResultEntity.date == data.get('date'),
                    PredictionResultEntity.type == data.get('type'),
                    PredictionResultEntity.version == data.get('version'),
                ).delete(synchronize_session=False)
            session.commit()
        finally:
            session.close()

    def get_by_date_and_type(self, date: str, type: str, prediction_limit: float) -> List[PredictionResultEntity]:
        """Retrieve all feature entities."""
        session = self.db_manager.Session()
        try:
            return session.query(PredictionResultEntity).filter(
                PredictionResultEntity.date == date,
                PredictionResultEntity.type == type,
                PredictionResultEntity.prediction >= prediction_limit
            ).all()
        finally:
            session.close()
  
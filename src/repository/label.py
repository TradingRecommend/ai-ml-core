from typing import List, Optional
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session
from src.config.constants import TradeType
from src.entities.label import LabelEntity
from src.database.db import DatabaseManager

class LabelRepository:
    def __init__(self):
        self.db_manager = DatabaseManager()
        LabelEntity.metadata.create_all(self.db_manager.get_engine())

    def save(self, label: LabelEntity) -> None:
        session = self.db_manager.Session()
        try:
            session.add(label)
            session.commit()
        finally:
            session.close()

    def save_multiple(self, labels: List[LabelEntity]) -> None:
        session = self.db_manager.Session()
        try:
            session.bulk_save_objects(labels)
            session.commit()
        finally:
            session.close()

    def get_by_symbol_and_date(self, symbol: str, date: str) -> Optional[LabelEntity]:
        session = self.db_manager.Session()
        try:
            return session.query(LabelEntity).filter_by(symbol=symbol, date=date).first()
        finally:
            session.close()

    def get_by_symbol(self, symbol: str) -> List[LabelEntity]:
        session = self.db_manager.Session()
        try:
            return session.query(LabelEntity).filter_by(symbol=symbol).all()
        finally:
            session.close()

    def get_by_type(self, type: str) -> List[LabelEntity]:
        session = self.db_manager.Session()
        try:
            return session.query(LabelEntity).filter_by(type=type).all()
        finally:
            session.close()

    def get_labels_not_in_stock_features(self) -> list[LabelEntity]:
        session = self.db_manager.Session()
        try:
            sql = """
                SELECT a.*
                FROM labels a
                WHERE a.type = :type
                AND (a.symbol || a.date || a.type) NOT IN (
                    SELECT (symbol || date || type)
                    FROM stock_features
                )
            """
            rows = session.execute(text(sql), {'type': TradeType.STOCK.value}).mappings().all()
            return [LabelEntity(**row) for row in rows]
        finally:
            session.close()

    def get_labels_not_in_top_coin_features(self) -> list[LabelEntity]:
        session = self.db_manager.Session()
        try:
            sql = """
                SELECT a.*
                FROM labels a
                WHERE a.type = :type
                AND (a.symbol || a.date || a.type) NOT IN (
                    SELECT (symbol || date || type)
                    FROM top_coin_features
                )
            """
            rows = session.execute(text(sql), {'type': TradeType.TOP_COIN.value}).mappings().all()
            return [LabelEntity(**row) for row in rows]
        finally:
            session.close()

    def get_all(self) -> List[LabelEntity]:
        session = self.db_manager.Session()
        try:
            return session.query(LabelEntity).all()
        finally:
            session.close()

    def update(self, label: LabelEntity) -> None:
        session = self.db_manager.Session()
        try:
            session.merge(label)
            session.commit()
        finally:
            session.close()

    def delete(self, symbol: str, date: str) -> None:
        session = self.db_manager.Session()
        try:
            label = session.query(LabelEntity).filter_by(symbol=symbol, date=date).first()
            if label:
                session.delete(label)
                session.commit()
        finally:
            session.close()
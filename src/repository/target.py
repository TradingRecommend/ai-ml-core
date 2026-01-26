from typing import List, Optional
from sqlalchemy.orm import Session
from src.database.db import DatabaseManager
from src.entities.target import TargetEntity

class TargetRepository:
    def __init__(self):
        self.db_manager = DatabaseManager()
        TargetEntity.metadata.create_all(self.db_manager.get_engine())

    def save(self, label: TargetEntity) -> None:
        session = self.db_manager.Session()
        try:
            session.add(label)
            session.commit()
        finally:
            session.close()

    def save_multiple(self, targets: List[TargetEntity]) -> None:
        session = self.db_manager.Session()
        try:
            session.bulk_save_objects(targets)
            session.commit()
        finally:
            session.close()


    def get_by_type(self, type: str) -> List[TargetEntity]:
        session = self.db_manager.Session()
        try:
            return session.query(TargetEntity).filter_by(type=type).all()
        finally:
            session.close()

    def get_all(self) -> List[TargetEntity]:
        session = self.db_manager.Session()
        try:
            return session.query(TargetEntity).all()
        finally:
            session.close()

    def update(self, target: TargetEntity) -> None:
        session = self.db_manager.Session()
        try:
            session.merge(target)
            session.commit()
        finally:
            session.close()

    def delete_by_type_value(self, type: str) -> None:
        session = self.db_manager.Session()
        try:
            session.query(TargetEntity).filter_by(type=type).delete()
            session.commit()
        finally:
            session.close()
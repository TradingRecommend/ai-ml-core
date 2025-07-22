# Base for ETL classes.
from abc import ABC, abstractmethod

class PredictionBase(ABC):
    @abstractmethod
    def run(self):
        pass
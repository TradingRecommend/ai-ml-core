# Base for ETL classes.
from abc import ABC, abstractmethod

class TrainMLModelBase(ABC):
    @abstractmethod
    def run(self):
        pass
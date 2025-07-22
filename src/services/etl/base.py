# Base for ETL classes.
from abc import ABC, abstractmethod

class ETLBase(ABC):
    """
    Base class for ETL (Extract, Transform, Load) operations.
    This class defines the structure for ETL processes.
    """

    @abstractmethod
    def extract(self):
        """
        Extract data from a source.
        """
        pass

    @abstractmethod
    def transform(self, data):
        """
        Transform the extracted data.
        :param data: The data to be transformed.
        """
        pass

    @abstractmethod
    def load(self, data):
        """
        Load the transformed data into a target system.
        :param data: The data to be loaded.
        """
        pass
    def run(self):
        """
        Run the ETL process.
        This method orchestrates the extract, transform, and load steps.
        """
        data = self.extract()
        transformed_data = self.transform(data)
        self.load(transformed_data)
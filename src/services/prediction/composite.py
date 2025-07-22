from typing import List

from src.services.prediction.base import PredictionBase



class PredictionComposite:
    """
    Composite class for ETL operations.
    This class is used to manage and execute multiple ETL operations in a composite manner.
    """

    def __init__(self):
        self.etl_operations: List[PredictionBase] = []

    def add_operation(self, operation):
        """
        Add an ETL operation to the composite.
        :param operation: An instance of an ETL operation.
        """
        self.etl_operations.append(operation)

    def run(self):
        """
        Run all ETL operations in the composite.
        """
        for operation in self.etl_operations:
            operation.run()
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class DatabaseManager:
    _instance = None
    Session: sessionmaker = None

    def __new__(cls):
        """Ensure only one instance of DatabaseManager is created."""
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            # Initialize the instance
            load_dotenv()  # Load environment variables from .env file
            db_user = os.getenv("DB_USER")
            db_password = os.getenv("DB_PASSWORD")
            db_host = os.getenv("DB_HOST")
            db_port = os.getenv("DB_PORT")
            db_name = os.getenv("DB_NAME")
            
            if not all([db_user, db_password, db_host, db_port, db_name]):
                raise ValueError("Missing required database connection parameters in .env file")
            
            connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            cls._instance.engine = create_engine(connection_string) 
            cls._instance.Session = sessionmaker(bind=cls._instance.engine)
        return cls._instance

    def get_engine(self):
        """Return the SQLAlchemy engine."""
        return self.engine

    def dispose(self):
        """Dispose of the database engine."""
        self.engine.dispose()
        # Optionally reset the instance to allow reinitialization if needed
        DatabaseManager._instance = None
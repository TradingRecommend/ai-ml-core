# Trading Recommend Project

## Table of Contents

1. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
2. [Folder Structure](#folder-structure)

## Getting Started

### Prerequisites

Before running the project, ensure you have the following software installed:

- [Python 3.10](https://www.python.org/)

### Installation

Follow these steps to install project dependencies:

```bash
# Build docker
docker build --platform=linux/arm64 -t ai-ml-core:v1 .

# Run project in local
PYTHONPATH=. python src/main.py train_model_job --trade-type TOP_COIN --model LOGISTIC

PYTHONPATH=. python src/main.py prediction_pipeline --trade-type TOP_COIN --model LOGISTIC --date 20250819

# Run project in docker
docker run -v $(pwd)/.prod.env:/app/.env --rm --name my-mlflow-container ai-ml-core:v1 train_model_job --trade-type STOCK --model LOGISTIC

docker run -v $(pwd)/.prod.env:/app/.env --rm --name my-mlflow-container ai-ml-core:v1 prediction_pipeline --trade-type STOCK --model LOGISTIC --date 20250819
```

## Folder Structure

The project follows the following folder structure:

    .
    ├── src                                     # Source files
    │   ├── cli                                 # Project command line
    │   ├── config                              # Project config
    │   ├── database                            # Database config
    │   ├── entities                            # Entities
    │   ├── repository                          # Repository
    │   ├── services                            # Services
    │   │   ├── *
    │   │   │   ├── etl                         # ETL services
    │   │   │   ├── training                    # Training services
    │   │   │   └── prediction                  # Prediction services
    │   ├── utils                               # Utilities functions
    ├── .env.example                            # Secret environment
    └── README.md

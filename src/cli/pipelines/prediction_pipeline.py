import click
from src.config.constants import ModelType, TradeType
from src.services.etl.composite import ETLComposite
from src.services.etl.stock.etl_prediction_feature import ETLPredictionStockFeature
from src.services.prediction.composite import PredictionComposite
from src.services.prediction.stock.stock_logistic_prediction import StockLogisticPrediction

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    '--trade-type',
    required=True,
    help='Trade type: ' + ', '.join([f"{e.name}" for e in TradeType])
)
@click.option(
    '--model', 
    required=True, 
    help='Comma-separated trade type: ' + ', '.join([f"{e.value} ({e.name.lower()})" for e in ModelType])
)
@click.option(
    '--date',
    required=True,
)
def prediction_pipeline(trade_type, model, date):
    composite_etl = ETLComposite()
    prediction_model = PredictionComposite()
    
    if trade_type == TradeType.STOCK.name:
        composite_etl.add_operation(ETLPredictionStockFeature(date=date)) 

        if model == ModelType.LOGISTIC.value:
            prediction_model.add_operation(StockLogisticPrediction(date=date)) 

    # Run the ETL composite
    composite_etl.run()

    # Run the prediction model
    prediction_model.run()

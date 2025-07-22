import click
from config.constants import ModelType, TradeType
from src.services.prediction.composite import PredictionComposite
from src.services.prediction.stock.stock_logistic_prediction import StockLogisticPrediction

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    '--trade-type',
    required=True,
    help='Comma-separated trade type: ' + ', '.join([f"{e.name}" for e in TradeType])
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
def prediction(trade_type, model, date):
    prediction_model = PredictionComposite()
    
    if trade_type == TradeType.STOCK.name and model == ModelType.LOGISTIC.value:
        prediction_model.add_operation(StockLogisticPrediction(date=date)) 

    prediction_model.run()
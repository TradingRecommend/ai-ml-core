import click
from config.constants import ModelType, TradeType
from src.services.training.composite import TrainMLModelComposite
from src.services.training.stock.stock_logistic import StockLogisticModel

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
def train_model(trade_type, model):
    composite_train_model = TrainMLModelComposite()
    
    if trade_type == TradeType.STOCK.name and model == ModelType.LOGISTIC.value:
        composite_train_model.add_operation(StockLogisticModel()) 

    composite_train_model.run()
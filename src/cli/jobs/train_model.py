import click
from config.constants import ModelType, TradeType
from src.services.training.composite import TrainMLModelComposite
from src.services.training.penny_coin.penny_coin_logistic import PennyCoinLogisticModel
from src.services.training.stock.stock_logistic import StockLogisticModel
from src.services.training.top_coin.top_coin_logistic import TopCoinLogisticModel
from src.services.training.stock.stock_decision_tree import StockDecisionTreeModel

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    '--trade-type',
    required=True,
    help='Comma-separated trade type: ' + ', '.join([f"{e.name}" for e in TradeType])
)
@click.option(
    '--model', 
    required=True, 
)
def train_model(trade_type, model):
    composite_train_model = TrainMLModelComposite()
    
    if trade_type == TradeType.STOCK.name and model == ModelType.LOGISTIC.value:
        composite_train_model.add_operation(StockLogisticModel()) 
    # support decision tree by explicit name "DECISION_TREE" (case-insensitive)
    if trade_type == TradeType.STOCK.name and model.upper() == ModelType.DECISION_TREE.value:
        composite_train_model.add_operation(StockDecisionTreeModel())
    if trade_type == TradeType.TOP_COIN.name and model == ModelType.LOGISTIC.value:
        composite_train_model.add_operation(TopCoinLogisticModel()) 
    if trade_type == TradeType.PENNY_COIN.name and model == ModelType.LOGISTIC.value:
        composite_train_model.add_operation(PennyCoinLogisticModel()) 

    composite_train_model.run()
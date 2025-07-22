import click
from src.config.constants import TradeType
from src.services.etl.composite import ETLComposite
from src.services.etl.stock.etl_training_feature import ETLTrainingStockFeature

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    '--trade-types',
    required=True,
    help='Comma-separated trade types: ' + ', '.join([f"{e.name}" for e in TradeType])
)
def etl_train_feature(trade_types):
    # Split and validate
    trade_type_list = [t.strip() for t in trade_types.split(',')]

    # Now you can use trade_type_list for your logic

    composite_etl = ETLComposite()
    
    for trade_type in trade_type_list:
        if trade_type == TradeType.STOCK.name:
            composite_etl.add_operation(ETLTrainingStockFeature()) 

    composite_etl.run()
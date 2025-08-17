import click
from src.config.constants import TradeType
from src.services.etl.coin.etl_prediction_feature import ETLPredictionTopCoinFeature
from src.services.etl.composite import ETLComposite
from src.services.etl.stock.etl_prediction_feature import ETLPredictionStockFeature

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    '--trade-types',
    required=True,
    help='Comma-separated trade types: ' + ', '.join([f"{e.name}" for e in TradeType])
)
@click.option(
    '--date',
    required=True,
)
def etl_prediction_feature(trade_types, date):
    # Split and validate
    trade_type_list = [t.strip() for t in trade_types.split(',')]

    # Now you can use trade_type_list for your logic

    composite_etl = ETLComposite()
    
    for trade_type in trade_type_list:
        if trade_type == TradeType.STOCK.name:
            composite_etl.add_operation(ETLPredictionStockFeature(date=date)) 
        elif trade_type == TradeType.TOP_COIN.name:
            composite_etl.add_operation(ETLPredictionTopCoinFeature(date=date))

    composite_etl.run()
import click
from src.services.etl.composite import ETLComposite
from src.services.etl.penny_coin.etl_target import ETLPennyCoinTarget

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
def etl_penny_coin_target():
    composite_etl = ETLComposite()
    
    composite_etl.add_operation(ETLPennyCoinTarget())

    composite_etl.run()
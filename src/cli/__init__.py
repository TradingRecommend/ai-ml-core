import click

from src.cli.jobs.etl_penny_coin_target import etl_penny_coin_target
from src.cli.jobs.etl_prediction_feature import etl_prediction_feature
from src.cli.jobs.etl_train_feature import etl_train_feature
from src.cli.jobs.prediction import prediction
from src.cli.jobs.train_model import train_model
from src.cli.pipelines.prediction_pipeline import prediction_pipeline


@click.group()
@click.version_option(version='2.4.2')
@click.pass_context
def cli(ctx):
    pass


# Jobs
cli.add_command(etl_prediction_feature, "etl_prediction_feature_job")
cli.add_command(etl_train_feature, "etl_train_feature_job")
cli.add_command(train_model, "train_model_job")
cli.add_command(prediction, "prediction_job")
cli.add_command(etl_penny_coin_target, "etl_penny_coin_target_job")

# Pipelines
cli.add_command(prediction_pipeline, "prediction_pipeline")
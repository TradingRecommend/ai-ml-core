import click

from .pipelines.prediction_pipeline import prediction_pipeline

from .jobs.train_model import train_model
from .jobs.prediction import prediction
from .jobs.etl_prediction_feature import etl_prediction_feature
from .jobs.etl_train_feature import etl_train_feature

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

# Pipelines
cli.add_command(prediction_pipeline, "prediction_pipeline")
from AI_Real_Classifier import logging
from AI_Real_Classifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from AI_Real_Classifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline

STAGE_NAME= "Data Ingestion stage"

try:
    logging.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    ingest= DataIngestionTrainingPipeline()
    ingest.main()
    logging.info(f">>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise e

STAGE_NAME= "Prepare Model"
try:
    logging.info(f"*******************")
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    prepare_model = PrepareBaseModelTrainingPipeline()
    prepare_model.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise e
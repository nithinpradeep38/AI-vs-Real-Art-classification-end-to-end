from AI_Real_Classifier import logging
from AI_Real_Classifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME= "Data Ingestion stage"

try:
    logging.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    ingest= DataIngestionTrainingPipeline()
    ingest.main()
    logging.info(f">>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise e

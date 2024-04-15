from AI_Real_Classifier.config.configuration import ConfigurationManager
from AI_Real_Classifier.components.data_ingestion import DataIngestion
from AI_Real_Classifier import logging

STAGE_NAME= "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config= ConfigurationManager() #initialize the configuration through ConfigurationManager class
            data_ingestion_config= config.get_data_ingestion_config() #get the data ingestion config of the config object
            data_ingestion= DataIngestion(config= data_ingestion_config) #define the component class for data ingestion
            data_ingestion.extract_zip_file()
        except Exception as e:
            raise e
        
if __name__ == '__main__':
    try:
        logging.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        ingest= DataIngestionTrainingPipeline()
        ingest.main()
        logging.info(f">>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
    
import zipfile
import os
from AI_Real_Classifier import logging
from AI_Real_Classifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig): 
        self.config= config #we get this configuration from the above get_data_ingestion_config

    def extract_zip_file(self):
        """
        zip_file_path:str
        extracts the zip file into the data directory
        function returns None
        """
        unzip_path= self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        logging.info(f"Extracted zip file to {unzip_path}")
        


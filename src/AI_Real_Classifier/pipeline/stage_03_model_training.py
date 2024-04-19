from AI_Real_Classifier.config.configuration import ConfigurationManager
from AI_Real_Classifier.components.model_training import Training
from AI_Real_Classifier import logging

STAGE_NAME= "Model Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config= config.get_training_config()
        training= Training(config= training_config)
        training.get_base_model()
        training.pre_process()
        training.train_valid_generator()
        training.train()



if __name__ == '__main__':
    try:
        logging.info(f"*******************")
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        train = ModelTrainingPipeline()
        train.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
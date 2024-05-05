from AI_Real_Classifier.config.configuration import ConfigurationManager
from AI_Real_Classifier.components.model_evaluation_with_mlflow import ModelEvaluation
from AI_Real_Classifier.components.model_training import Training
from AI_Real_Classifier import logging

STAGE_NAME= "Model Evaluation with MLFlow"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config= ConfigurationManager()
            training_config= config.get_training_config()
            training= Training(config= training_config)
            training.pre_process()
            training.train_valid_generator()
            eval_config= config.get_evaluation_config()
            evaluation= ModelEvaluation(validation_generator=training.validation_generator, config=eval_config)
            evaluation.evaluation()
            #commenting out the below since model is finalized and we are deploying to production.
            #evaluation.log_into_mlflow() 

        except Exception as e:
            raise e
        
if __name__ == '__main__':
    try:
        logging.info(f"*******************")
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        evaluate = ModelEvaluationPipeline()
        evaluate.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
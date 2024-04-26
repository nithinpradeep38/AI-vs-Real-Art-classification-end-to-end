import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from AI_Real_Classifier.entity.config_entity import EvaluationConfig
from AI_Real_Classifier.utils.common import save_json

class ModelEvaluation:
    def __init__(self,validation_generator, config: EvaluationConfig):
        self.config= config
        self.validation_generator= validation_generator
        

    @staticmethod
    def load_model(path: Path)-> tf.keras.Model:
        return tf.keras.models.load_model(path)
    
    def save_score(self):
        scores= {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path= Path("scores.json"), data= scores)

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri) #set the registered uri
        tracking_url_type_store= urlparse(mlflow.get_tracking_uri()).scheme #checks if is it file or remote server (http)

        with mlflow.start_run() as run_id:
            mlflow.log_params(self.config.all_params) #first log all the parameters from config
            mlflow.log_metrics(     
                {"loss": self.score[0],
                 "accuracy": self.score[1]}             #log the result from evaluation on validation data 
            )

            #model registry does not work with file store
            if tracking_url_type_store != "file":

                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model") #track in registered uri

            else: 
                mlflow.keras.log_model(self.model, "model") #track in local system

    
    def evaluation(self):
        self.model= self.load_model(self.config.path_of_model)
        self.score= self.model.evaluate(self.validation_generator)
        self.save_score()
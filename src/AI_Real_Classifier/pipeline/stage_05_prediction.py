import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import os

class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename



    def predict(self):
        # load model
        model = load_model(os.path.join("model", "model.h5"))
        class_labels = ['AI Generated', 'Real']
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        test_image= preprocess_input(test_image)
        test_image /= 255.0
        
        predictions= model.predict(test_image)
        predicted_class= class_labels[np.argmax(predictions)]

        print(predicted_class)
        print(predictions)
        
        return [{ "image" : predicted_class}]
        
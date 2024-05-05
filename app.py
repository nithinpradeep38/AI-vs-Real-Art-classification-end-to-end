from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from AI_Real_Classifier.utils.common import decodeImage
from AI_Real_Classifier.pipeline.stage_05_prediction import PredictionPipeline

app= Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename= "inputImage.jpg" #the image from user
        self.classifier= PredictionPipeline(self.filename) #calling the pipeline for prediction



@app.route("/", methods= ['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py") #run main.py for training
    # os.system("dvc repro")    #or the dvc repro command
    return "Training done successfully!"

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, ClientApp.filename)
    result = ClientApp.classifier.predict()
    return jsonify(result)

if __name__ == "__main__":
    ClientApp= ClientApp()
    app.run(host='0.0.0.0', port=8080) #local host
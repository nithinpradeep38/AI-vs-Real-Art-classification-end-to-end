# AI-vs-Real-Art-classification-end-to-end
An end-to-end production grade CNN based project which attempts to classify the art as AI generated or Real Art. 
I have used a VGG16 based CNN model with a transfer learning approach.

The detailed experimentation can be found at https://github.com/nithinpradeep38/AI-Vs-Real-Art-Classification 

I have used the following tools for MLOps

| Tool         | Description                                     |
|--------------|-------------------------------------------------|
| MLFlow       | Experiment tracking platform for machine learning models |
| DVC          | Data version control system                     |
| Docker       | Containerization platform for packaging code and dependencies |
| Github Actions | Continuous integration and continuous deployment platform |
| AWS ECR |  Docker Container registry service provided by Amazon Web Services |
| AWS EC2      | Virtual machine service on Amazon Web Services for deployment |



Note: I have disabled the AWS after testing the deployment so as to not incur charges. Below is a screenshot of the public IP address of the app.
<img width="1897" alt="328074016-d77df94f-d0a5-4d01-96a6-f06d975bc065" src="https://github.com/nithinpradeep38/AI-vs-Real-Art-classification-end-to-end/assets/96964974/6d0bc4d1-f72d-4c17-8500-64d477ab010a">

Alternatively, this can be downloaded and run locally by running \
```python app.py```

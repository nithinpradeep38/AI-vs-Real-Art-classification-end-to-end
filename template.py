import os
from pathlib import Path
import logging

logging.basicConfig(level= logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name= 'AI_Real_Classifier'

#list of files we need in the template
list_of_files = [
    ".github/workflows/.gitkeep",                            #for github actions for CI-CD. We will have the main.yaml file here
    f"src/{project_name}/__init__.py",                       #constructor file for creating local package
    f"src/{project_name}/components/__init__.py",   
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",                                   
    "dvc.yaml",                                              #for DVC integration
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",                                
    "templates/index.html"
                ]


for filepath in list_of_files:
    filepath= Path(filepath)        #OS agnostic way of creating valid path
    filedir, filename= os.path.split(filepath)      

    #creating folder directory
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory:{filedir} for the filename: {filename}")

    #creating files
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"File {filename} already exists at: {filepath}")
    
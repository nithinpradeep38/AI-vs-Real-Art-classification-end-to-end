import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
import random
from AI_Real_Classifier import logging
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from keras.callbacks import ModelCheckpoint
from PIL import Image
from pathlib import Path
from AI_Real_Classifier.entity.config_entity import TrainingConfig

class Training:
    def __init__(self, config: TrainingConfig):
        self.config= config
    
    def get_base_model(self):
        self.model= tf.keras.models.load_model(self.config.updated_base_model_path)

    @staticmethod
    def sample_fake_images(folder, num_samples):
        random.seed(42)
        return random.sample(os.listdir(folder), num_samples)

    @staticmethod
    def filter_large_images(image_dir, max_pixels=178956970):
            filtered_images = []
            for filename in os.listdir(image_dir):
                filepath = os.path.join(image_dir, filename)
                try:
                    with Image.open(filepath) as img:
                        if img.size[0] * img.size[1] <= max_pixels:
                            continue
                            #filtered_images.append(filename)
                except Exception as e:
                    logging.info(f"Error processing {filename}: {e}")
                    filtered_images.append(filename)
            return filtered_images

    @staticmethod
    def load_filenames_labels(folder, label, large_img, sampled_imgs=None):
        if sampled_imgs is None:
            sampled_imgs= os.listdir(folder)
        filenames = []
        labels = []
        for filename in os.listdir(folder):
            if (filename not in large_img) and (filename in sampled_imgs) :
                filenames.append(os.path.join(folder, filename))
                labels.append(label)
        return filenames, labels
    
    def pre_process(self):
        self.filtered_images_fake= self.filter_large_images(self.config.class_fake)
        self.filtered_images_real= self.filter_large_images(self.config.class_real)
        self.sampled_fake= self.sample_fake_images(self.config.class_fake,4000)
        self.class_fake_filenames, self.class_fake_labels= self.load_filenames_labels(self.config.class_fake, 
                                                                                      '0', 
                                                                                      self.filtered_images_fake,
                                                                                      self.sampled_fake)
        
        self.class_real_filenames, self.class_real_labels= self.load_filenames_labels(self.config.class_real, 
                                                                                      '1', 
                                                                                      self.filtered_images_real)
        
        self.all_file_names= self.class_fake_filenames+ self.class_real_filenames
        self.all_labels= self.class_fake_labels+ self.class_real_labels


        # Split the data into train and validation sets while maintaining class balance
        self.train_filenames, self.validation_filenames, self.train_labels, self.validation_labels = train_test_split(
            self.all_file_names, self.all_labels, test_size=0.2, stratify=self.all_labels, random_state=42)
    
    def train_valid_generator(self):
        
        self.train_datagen= ImageDataGenerator(rescale=1./255, preprocessing_function= preprocess_input)
        self.valid_datagen= ImageDataGenerator(rescale=1./255, preprocessing_function= preprocess_input)

    # Create the generator for training data
        self.train_generator = self.train_datagen.flow_from_dataframe(
            dataframe=pd.DataFrame({'filename': self.train_filenames, 'class': self.train_labels}),
            x_col='filename',
            y_col='class',
            target_size=(224,224),
            batch_size=self.config.params_batch_size,
            class_mode='categorical',
            shuffle=False,
            seed=42
        )

        # Create the generator for validation data
        self.validation_generator = self.valid_datagen.flow_from_dataframe(
            dataframe=pd.DataFrame({'filename': self.validation_filenames, 'class': self.validation_labels}),
            x_col='filename',
            y_col='class',
            target_size=(224,224),
            batch_size=self.config.params_batch_size,
            class_mode='categorical',
            shuffle=False,
            seed=42
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        self.filepath= str(self.config.trained_model_path)
        self.checkpoint= ModelCheckpoint(filepath= self.filepath, 
                                         monitor="val_accuracy",
                                         verbose=1,
                                         save_best_only=True,
                                         mode='max')
        self.model.fit(self.train_generator,
                       steps_per_epoch=len(self.train_generator),
                       epochs= self.config.params_epochs,
                       validation_data= self.validation_generator,
                       validation_steps= len(self.validation_generator),
                       callbacks= [self.checkpoint])
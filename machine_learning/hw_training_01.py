# print('hello world')
import os
import azureml
import glob
from zipfile import ZipFile
# from hw_training_prep import create_train_validate_test_sets, normalize_images, build_CNN_model, load_images_subset_and_shuffle
from hw_training_prep import load_images, normalize_images, build_CNN_model, download_dataset_of_images, load_images_subset_and_shuffle

from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import Workspace, Datastore, Dataset, Run
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.dataset import Dataset
import tempfile

# mount_context.start()
import numpy as np
import pandas as pd
import cv2
import time
import json
from glob import glob
# from matplotlib import pyplot as plt
# %matplotlib inline

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import initializers

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print('all imports successful')
print('tf version', tf.__version__)

# class Heatwave_Model_Training:
    
#     def __init__(self, data_folder_name, images_folder_name, data_file_name, train_pct, validate_pct):
#         self.data_folder_name = data_folder_name
#         self.images_folder_name = images_folder_name
#         self.data_file_name = data_file_name
#         self.train_pct = train_pct
#         self.validate_pct = validate_pct

#         self.train_X, self.train_Y, self.train_X_names \
#         , self.validate_X, self.validate_Y, self.validate_X_names \
#         , self.test_X, self.test_Y, self.test_X_names = [None]*9

#         self.layers_to_add = []

#     def create_normalized_subsets(self, limit=0):
#         """
#         If limit is not 0, the specified number of images will be considered. Useful for sampling images and testing logic.
#         """
#         self.train_X, self.train_Y, self.train_X_names, self.validate_X, self.validate_Y, self.validate_X_names \
#         , self.test_X, self.test_Y, self.test_X_names = create_train_validate_test_sets(self.data_folder_name, self.images_folder_name
#                                                                                 , self.data_file_name, self.train_pct
#                                                                                 , self.validate_pct, limit=limit)

#         #### normalize images
#         self.train_X, self.validate_X, self.test_X = normalize_images(self.train_X, self.validate_X, self.test_X)

#     def reset_layers(self):
#         #### prepare for creating the model
#         self.layers_to_add = []

#     def add_layers(self, layerType='D', filters=None, kernel_size=None, pool_size=None, strides=None):
#       """
#       LayerType   : 'C' for Convolution, 'D' for Dense, 'M' for Maxpool
#       Filters     : Number of filters, or None
#       kernel_size : (x, y) tuple, as applicable to Convolution, or None.
#       pool_size   : (x, y) tuple, as applicable to Maxpool, or None.
#       strides     : (x, y) tuple, as applicable to Convolution or Maxpool, or None.
#       """
#       validation_passed = True

#       if layerType not in ['C', 'D', 'M']:
#         print("Validation error: layer type must be 'C' for Convolution, 'D' for Dense, or 'M' for Maxpool")
#         validation_passed = False

#       layerParams = {'layerType': layerType}
#       if filters is not None:
#           layerParams['filters'] = filters
#       if kernel_size is not None:
#           layerParams['kernel_size'] = kernel_size
#       if pool_size is not None:
#           layerParams['pool_size'] = pool_size
#       if strides is not None:
#           layerParams['strides'] = strides
#       self.layers_to_add.append(layerParams)
        

#     def create_model(self, num_classes = 1):
#         print('Creating model...')
#         #### num_classes = 1 means binary classification
#         #### input_shape = [image_height, image_width, num_channels] # height, width, channels
#         input_shape = [int(np.shape(self.train_X)[1]), int(np.shape(self.train_X)[2]), int(np.shape(self.train_X)[3])] 
#         print('input_shape =', input_shape, 'num_classes =', num_classes)
#         model = build_CNN_model(input_shape, num_classes, self.layers_to_add)

#         # Print the model summary
#         print(model.summary())
#         return model


#     def prepare_model_train_01_image_only(self):
#         print('Preparing model for training 01 -- images only...')
#         self.reset_layers()
#         # layer 1
#         self.add_layers(layerType='C', filters=16, kernel_size= (3,3), pool_size=None, strides=None)        
#         # layer 2
#         self.add_layers(layerType='M', pool_size=(2,2), strides=(2,2))
#         # layer 3
#         self.add_layers(layerType='C', filters=32, kernel_size= (3,3))                
#         return self.create_model()


#     def prepare_model_train_02_image_only(self):
#         print('Preparing model for training 01 -- images only...')
#         self.reset_layers()
#         # layer 1
#         self.add_layers(layerType='C', filters=16, kernel_size= (3,3), pool_size=None, strides=None)        
#         # layer 2
#         self.add_layers(layerType='M', pool_size=(2,2), strides=(2,2))
#         # layer 3
#         self.add_layers(layerType='C', filters=32, kernel_size= (3,3))                
#         # layer 4
#         self.add_layers(layerType='M', pool_size=(2,2), strides=(2,2))
#         # layer 5
#         self.add_layers(layerType='C', filters=64, kernel_size= (3,3))                
#         return self.create_model()


#     def train_binary_classification_model(self, model, learning_rate, epochs, patience):
#         #### Prepare to train
#         # Free up memory
#         # K.clear_session()

#         # Optimizer
#         optimizer = optimizers.SGD(lr=learning_rate)

#         # Loss
#         loss = losses.binary_crossentropy

#         # Compile
#         model.compile(loss=loss,
#                         optimizer=optimizer,
#                         metrics=['accuracy'])

#         # using Early Stopping to stop in case of overfitting... 
#         es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)

#         #### Train model
#         start_time = time.time()
#         training_results = model.fit(
#                 self.train_X, self.train_Y, 
#                 validation_data=(self.validate_X, self.validate_Y),
#                 epochs=epochs, 
#                 verbose=1,
#                 callbacks=es)
#         execution_time = (time.time() - start_time)/60.0
#         print("Training execution time (mins)",round(execution_time, 2))

#         #### Evaluate on Test data
#         results = model.evaluate(self.test_X, self.test_Y)
#         print("test loss = {}, test acc = {}".format(round(results[0], 2), round(results[1], 2)))


# def load_dataset_as_normalized(limit=0):
#     data_folder_name = 'data'
#     images_folder_name = 'images'
#     data_file_name = '1980_to_2030__hw_area_pct_30_labels.csv'
#     train_pct = 0.7
#     validate_pct = 0.15

#     hmt = Heatwave_Model_Training(data_folder_name, images_folder_name, data_file_name, train_pct, validate_pct)
    
#     hmt.create_normalized_subsets(limit=limit)
#     return hmt

# def execute_training_01(hmt):     # Heatwave_Model_Training class object
#     print('*** execute_training_01 ***') 
#     model = hmt.prepare_model_train_01_image_only()
#     learning_rate = 0.01
#     epochs = 35
#     patience=5

#     hmt.train_binary_classification_model(model, learning_rate, epochs, patience)

# def execute_training_02(hmt):     # Heatwave_Model_Training class object  
#     print('*** execute_training_02 ***') 
#     model = hmt.prepare_model_train_02_image_only()
#     learning_rate = 0.01
#     epochs = 35
#     patience=5

#     hmt.train_binary_classification_model(model, learning_rate, epochs, patience)

# def experiment_with_dataset(ws, datastore_name, dataset_name):
#     mounted_path = tempfile.mkdtemp()
#     mount_context = dataset.mount(mounted_path)

#     # datastore = Datastore.get(ws, datastore_name)
    
#     ds = Dataset.get_by_name(workspace, name, version='latest')

# def main():
#     #### experiment using dataset
#     ws = Workspace.from_config()
#     sub_folder_name = 'aoi_107_2021_11_11'
#     datastore_name = 'workspaceblobstore'
#     # upds =  
#     # dataset_name = sub_folder_name + '_' + upds
#     print('hw_training_01.py ::: script running! workspace:\n', ws)

#     #### Run training
#     # limit = 40   # 40 for sampling and testing code
#     # hmt = load_dataset_as_normalized(limit)
#     # execute_training_01(hmt)
#     # execute_training_02(hmt)

# # main()


# def load_dataset_as_normalized(limit=0):
#     data_folder_name = 'data'
#     images_folder_name = 'images'
#     data_file_name = '1980_to_2030__hw_area_pct_30_labels.csv'
#     train_pct = 0.7
#     validate_pct = 0.15

#     hmt = Heatwave_Model_Training(data_folder_name, images_folder_name, data_file_name, train_pct, validate_pct)
    
#     hmt.create_normalized_subsets(limit=limit)
#     return hmt

# def execute_training_01(hmt):     # Heatwave_Model_Training class object
#     print('*** execute_training_01 ***') 
#     model = hmt.prepare_model_train_01_image_only()
#     learning_rate = 0.01
#     epochs = 35
#     patience=5

#     hmt.train_binary_classification_model(model, learning_rate, epochs, patience)

# def execute_training_02(hmt):     # Heatwave_Model_Training class object  
#     print('*** execute_training_02 ***') 
#     model = hmt.prepare_model_train_02_image_only()
#     learning_rate = 0.01
#     epochs = 35
#     patience=5

#     hmt.train_binary_classification_model(model, learning_rate, epochs, patience)


class LayerDetails:
    def __init__(self, layerType, filters, kernel_size, pool_size, strides):
        self.layerType = layerType
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.strides = strides

class Heatwave_Model_Training:
    
    def __init__(self, images_folder_name):
        self.images_folder_name = images_folder_name
        
        self.train_X, self.train_Y, self.train_X_names \
        , self.validate_X, self.validate_Y, self.validate_X_names \
        , self.test_X, self.test_Y, self.test_X_names = [None]*9

        self.layers_to_add = []

    def load_subsets_shuffle_and_normalize(self, limit=0):
        """
        If limit is not 0, the specified number of images will be considered. Useful for sampling images and testing logic.
        """
        # train
        subset_folder = os.path.join(self.images_folder_name, 'train')
        self.train_X, self.train_Y, self.train_X_names = load_images_subset_and_shuffle(subset_folder, limit)

        # validate
        subset_folder = os.path.join(self.images_folder_name, 'validate')
        self.validate_X, self.validate_Y, self.validate_X_names = load_images_subset_and_shuffle(subset_folder, limit)

        # test
        subset_folder = os.path.join(self.images_folder_name, 'test')
        self.test_X, self.test_Y, self.test_X_names = load_images_subset_and_shuffle(subset_folder, limit)

        
        #### normalize images
        self.train_X, self.validate_X, self.test_X = normalize_images(self.train_X, self.validate_X, self.test_X)

    
    def print_dataset_summary(self):
        """ Print some counts and details of the loaded dataset """
        print("Count of images = ", len(self.train_X))
        lbls, cnts = np.unique(self.train_Y, return_counts=True)
        for lblcnt in zip(lbls, cnts):
            print("Label {} -- count {}".format(lblcnt[0], lblcnt[1]))

        print('len(train_X)', len(self.train_X), 'len(train_Y)', len(self.train_Y), 'len(train_X_names)', len(self.train_X_names))
        print(np.shape(self.train_X[0]), self.train_Y[0], self.train_X_names[0])
        print(np.shape(self.train_X[1]), self.train_Y[1], self.train_X_names[1])
        print('')
        print('len(validate_X)', len(self.validate_X))
        print('len(test_X)', len(self.test_X))

    def reset_layers(self):
        #### prepare for creating the model
        self.layers_to_add = []
    

    def add_layers(self, layerType='D', filters=None, kernel_size=None, pool_size=None, strides=None):
      """
      LayerType   : 'C' for Convolution, 'D' for Dense, 'M' for Maxpool
      Filters     : Number of filters, or None
      kernel_size : (x, y) tuple, as applicable to Convolution, or None.
      pool_size   : (x, y) tuple, as applicable to Maxpool, or None.
      strides     : (x, y) tuple, as applicable to Convolution or Maxpool, or None.
      """
      validation_passed = True

      if layerType not in ['C', 'D', 'M']:
        print("Validation error: layer type must be 'C' for Convolution, 'D' for Dense, or 'M' for Maxpool")
        validation_passed = False

      layerParams = {'layerType': layerType}
      if filters is not None:
          layerParams['filters'] = filters
      if kernel_size is not None:
          layerParams['kernel_size'] = kernel_size
      if pool_size is not None:
          layerParams['pool_size'] = pool_size
      if strides is not None:
          layerParams['strides'] = strides
      self.layers_to_add.append(layerParams)
        

    def create_model(self, num_classes = 1):
        print('Creating model...')
        #### num_classes = 1 means binary classification
        #### input_shape = [image_height, image_width, num_channels] # height, width, channels
        input_shape = [int(np.shape(self.train_X)[1]), int(np.shape(self.train_X)[2]), int(np.shape(self.train_X)[3])] 
        print('input_shape =', input_shape, 'num_classes =', num_classes)
        model = build_CNN_model(input_shape, num_classes, self.layers_to_add)

        # Print the model summary
        print(model.summary())
        return model


    def prepare_model(self, layers):
        print('Preparing model for training...')
        self.reset_layers()
        for layer in layers:
            self.add_layers(layerType=layer.layerType, filters=layer.filters
                            , kernel_size= layer.kernel_size, pool_size=layer.pool_size, strides=layer.strides)        
        
        return self.create_model()

    
    def train_binary_classification_model(self, model, learning_rate, epochs, patience):
        #### Prepare to train
        # Free up memory
        # K.clear_session()

        # Optimizer
        optimizer = optimizers.SGD(lr=learning_rate)

        # Loss
        loss = losses.binary_crossentropy

        # Compile
        model.compile(loss=loss,
                        optimizer=optimizer,
                        metrics=['accuracy'])

        # using Early Stopping to stop in case of overfitting... 
        es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)

        #### Train model
        start_time = time.time()
        training_results = model.fit(
                self.train_X, self.train_Y, 
                validation_data=(self.validate_X, self.validate_Y),
                epochs=epochs, 
                verbose=1,
                callbacks=[es])
        # training_results = model.fit(
        #         self.train_X, self.train_Y, 
        #         validation_data=(self.validate_X, self.validate_Y),
        #         epochs=epochs, 
        #         verbose=1)
        execution_time = (time.time() - start_time)/60.0
        print("Training execution time (mins)",round(execution_time, 2))

        #### Evaluate on Test data
        results = model.evaluate(self.test_X, self.test_Y)
        print("test loss = {}, test acc = {}".format(round(results[0], 2), round(results[1], 2)))


def get_images_dataset_to_local(dataset_prefix):
    download_dataset_of_images(dataset_prefix)
    
def load_dataset_as_normalized(dataset_prefix, limit=0):
    hmt = Heatwave_Model_Training(images_folder_name=dataset_prefix)
    hmt.load_subsets_shuffle_and_normalize(limit)
    hmt.print_dataset_summary()

    return hmt

def execute_training_01(hmt):     # Heatwave_Model_Training class object
    print('*** execute_training_01 ***') 
    # layers
    build_with_layers = []
    build_with_layers.append(LayerDetails(layerType='C', filters=16, kernel_size= (3,3), pool_size=None, strides=None))
    build_with_layers.append(LayerDetails(layerType='M', filters=None, kernel_size= None, pool_size=(2,2), strides=(2,2)))
    build_with_layers.append(LayerDetails(layerType='C', filters=32, kernel_size= (3,3), pool_size=None, strides=None))               

    model = hmt.prepare_model(build_with_layers)
    learning_rate = 0.01
    epochs = 35
    patience=5
    print(model, '\n')

    hmt.train_binary_classification_model(model, learning_rate, epochs, patience)

def execute_training_02(hmt):     # Heatwave_Model_Training class object  
    print('*** execute_training_02 ***') 
    # layers
    build_with_layers = []
    build_with_layers.append(LayerDetails(layerType='C', filters=16, kernel_size= (3,3), pool_size=None, strides=None))
    build_with_layers.append(LayerDetails(layerType='M', filters=None, kernel_size= None, pool_size=(2,2), strides=(2,2)))
    build_with_layers.append(LayerDetails(layerType='C', filters=32, kernel_size= (3,3), pool_size=None, strides=None))               
    build_with_layers.append(LayerDetails(layerType='M', filters=None, kernel_size= None, pool_size=(2,2), strides=(2,2)))
    build_with_layers.append(LayerDetails(layerType='C', filters=64, kernel_size= (3,3), pool_size=None, strides=None))    

    model = hmt.prepare_model(build_with_layers)
    learning_rate = 0.01
    epochs = 35
    patience=5
    print(model, '\n')

    hmt.train_binary_classification_model(model, learning_rate, epochs, patience)


def main():
    #### experiment using dataset
    ws = Workspace.from_config()
    sub_folder_name = 'aoi_107_2021_11_11'
    dataset_prefix = 'aoi_107_2021_11_11'
    datastore_name = 'workspaceblobstore'
    print('hw_training_01.py ::: script running! workspace:\n', ws)
    
    print('hw_training_01.py ::: calling get_images_dataset_to_local()...')    
    get_images_dataset_to_local(dataset_prefix)
    
    limit = 40   # 40 for sampling and testing code, 0 for no limits
    
    print('hw_training_01.py ::: create Heatwave Model Training class object...')    
    hmt = load_dataset_as_normalized(dataset_prefix, limit)

    #### Run training    
    print('\nhw_training_01.py ::: executing training setup 01...')    
    execute_training_01(hmt)
    print('\nhw_training_01.py ::: executing training setup 02...')    
    execute_training_02(hmt)

# main()

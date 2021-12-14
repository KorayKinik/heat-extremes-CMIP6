# print('hello world')
import os
import azureml
import glob
import numpy as np
import pandas as pd
import time
import cv2
import shutil
from zipfile import ZipFile
from custom_utils import delete_all_files_in_the_folder

from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import Workspace, Run
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Datastore, Dataset
from azureml.data.datapath import DataPath

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

ws = Workspace.from_config()

print('ws :', ws)

def get_dataset_to_local(dataset_name, data_folder_name, images_folder_name):
    dataset = Dataset.get_by_name(ws, dataset_name)

    ### data folder
    data_folder = os.path.join(os.getcwd(), data_folder_name)
    os.makedirs(data_folder, exist_ok=True)

    zip_file_name = None
    for filename in os.listdir(data_folder):
        if filename.endswith('.zip'):
            zip_file_name = filename

    if zip_file_name is None:
        print('no zip file, downloading from dataset')
        dataset.download(data_folder, overwrite=True)    
        for filename in os.listdir(data_folder):    # check again for the zip file
            if filename.endswith('.zip'):
                zip_file_name = filename

    if zip_file_name is None:
        print('Still no zip file to work with. Stopping execution!')
    else:
        print('zip_file_name', zip_file_name)
        ### images folder
        images_folder = os.path.join(data_folder, images_folder_name)
        if os.path.exists(images_folder):
            response = input('images folder {} already exists.\n enter y (yes) to delete existing files and unzip,\n else n (no) to stop execution. : '.format(images_folder_name))
            if response.lower()[0] == 'y':
                delete_all_files_in_the_folder(images_folder)
                print('deleted files in the images folder, if any. Unzipping from images zip file {}'.format(zip_file_name))
                with ZipFile(os.path.join(data_folder, zip_file_name), 'r') as zip_ref:
                    zip_ref.extractall(images_folder)
                print('files extracted!')
            else:
                print('Response {}, is not y -- stopping execution!'.format(response))
        else:
            print('making the images folder')
            os.makedirs(images_folder, exist_ok=False)
            print('Unzipping from images zip file {}'.format(zip_file_name))

#### Helper functions:
def validations_for_create_train_validate_test_sets(data_folder, images_folder, data_file, train_pct, validate_pct):
    retval = True    
    if not os.path.exists(data_folder):
        print('Validation Failed: Data folder {} does not exist.'.format(data_folder))
        return False
    if not os.path.exists(images_folder):
        print('Validation Failed: Images folder {} does not exist.'.format(images_folder))
        return False
    if not os.path.exists(data_file):
        print('Validation Failed: Data file {} does not exist.'.format(data_file))
        return False
    if train_pct < 0.0 or train_pct > 1.0:
        print('Validation Failed: train_pct {} must be between 0.0 and 0.1'.format(train_pct))
        return False
    if validate_pct < 0.0 or validate_pct > 1.0:
        print('Validation Failed: validate_pct {} must be between 0.0 and 0.1'.format(validate_pct))
        return False
    if (train_pct + validate_pct) < 0.0 or (train_pct + validate_pct) > 1.0:
        print('Validation Failed: train_pct + validate_pct {}+{}={} must be between 0.0 and 0.1'.format(train_pct
                                                                                                        ,validate_pct
                                                                                                        ,train_pct + validate_pct))
        return False

    return retval

# A function that loads the images into numpy array - for the specified file names
def load_images(images_folder, file_names):
    """
    The function will load images for the specified folder and file names
    returns: np arrays X
    """
    X = []
    start = time.time()  
    start500 = time.time()  

    if file_names is None:
        file_names = [f for f in os.listdir(images_folder) if f.endswith('.jpg') or f.endswith('.png')]
    print('loading {} images...'.format(len(file_names)))
    i = 0
    for filename in file_names:        
        image = cv2.imread(os.path.join(images_folder,filename))  # read image        
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to rgb
        X.append(image)
        i += 1
        if i%500 == 0:
            print('{} images, and continuing...'.format(i))
            end = time.time()  
            print(' ... time taken for 500 images: ', round(end - start500,4))
            start500 = time.time()  

    print('Total {} images loaded'.format(i))
    end = time.time()  
    print('Total time taken: {}'.format(round(end - start,4)))

    # convert to numpy arrays and return
    X = np.asarray(X)    
    return X

# A function that loads the image names into and loads labels into numpy arrays
def load_labels_and_image_names(images_folder, data_file, limit=0):
    """
    The function will load image names and labels
    and if a limit is specified (for verfication steps), only that many rows will be processed
    Returns:
    tuple of np arrays X_names, and Y
    """
    X_names = np.array(os.listdir(images_folder))
    Y = np.genfromtxt(data_file, delimiter=',')
    
    if limit:
        X_names = X_names[:limit]
        Y = Y[:limit]

    return (X_names, Y)

def split_indices_proportionate_1s_into_train_validate_test(Y, train_pct, validate_pct, test_pct):
    """
    Function takes the labels np array. Distributes 1s proportionately while splitting indices into
    train, validate and test sets. 
    """
    # Split the 1s
    labeled_1_pct = np.round(np.mean(Y) * 100, 1)
    print('{}% images are labeled 1. Count: {}'.format(labeled_1_pct, np.sum(Y)))

    indices_of_1s = np.where(Y == 1)[0]
    print('len(indices_of_1s) =', len(indices_of_1s), ' indices_of_1s[:3] = ', indices_of_1s[:3])
    indices_of_0s = np.where(Y == 0)[0]
    print('len(indices_of_0s) =', len(indices_of_0s), ' indices_of_0s[:3] = ', indices_of_0s[:3])
    
    train_indices = np.random.choice(indices_of_1s, size=int(len(indices_of_1s)*train_pct), replace=False)
    test_indices = np.setdiff1d(indices_of_1s, train_indices)  # remaining, for now
    validate_indices = np.random.choice(test_indices, size=int(len(indices_of_1s)*validate_pct), replace=False)
    test_indices = np.setdiff1d(test_indices, validate_indices)  # remaining, finally
    
    print('len(train_indices) = {}, that is {:.1f} %'.format(len(train_indices), (len(train_indices)*100/len(indices_of_1s))))
    print('len(validate_indices) = {}, that is {:.1f} %'.format(len(validate_indices), (len(validate_indices)*100/len(indices_of_1s))))
    print('len(test_indices) = {}, that is {:.1f} %'.format(len(test_indices), (len(test_indices)*100/len(indices_of_1s))))

    # Split the 0s
    train_indices_0s = np.random.choice(indices_of_0s, size=int(len(indices_of_0s)*train_pct), replace=False)
    test_indices_0s = np.setdiff1d(indices_of_0s, train_indices_0s)  # remaining, for now
    validate_indices_0s = np.random.choice(test_indices_0s, size=int(len(indices_of_0s)*validate_pct), replace=False)
    test_indices_0s = np.setdiff1d(test_indices_0s, validate_indices_0s)  # remaining, finally
    
    print('len(train_indices_0s) = {}, that is {:.1f} %'.format(len(train_indices_0s), (len(train_indices_0s)*100/len(indices_of_0s))))
    print('len(validate_indices_0s) = {}, that is {:.1f} %'.format(len(validate_indices_0s), (len(validate_indices_0s)*100/len(indices_of_0s))))
    print('len(test_indices_0s) = {}, that is {:.1f} %'.format(len(test_indices_0s), (len(test_indices_0s)*100/len(indices_of_0s))))

    # Merge 0s with 1s -- and shuffle
    train_indices = np.concatenate((train_indices, train_indices_0s), axis=None)
    np.random.shuffle(train_indices)
    validate_indices = np.concatenate((validate_indices, validate_indices_0s), axis=None)
    np.random.shuffle(validate_indices)
    test_indices = np.concatenate((test_indices, test_indices_0s), axis=None)
    np.random.shuffle(test_indices)

    print('Sum of lenths of train, validate, test = ', len(train_indices) + len(validate_indices) + len(test_indices))

    return train_indices, validate_indices, test_indices

def save_train_validate_test_sets(data_folder_name, images_folder_name, data_file_name, sub_folder_name, train_pct, validate_pct, limit=0):
    """
    For the provided folder of images and data file name -- prepares the train, validate, test sets -- and stores them
    complete with X : images,  Y: labels, X_names: Additional_info (names of the files)
    """
    data_folder = os.path.join(os.getcwd(), data_folder_name)
    images_folder = os.path.join(data_folder, images_folder_name)
    sub_folder = os.path.join(os.path.dirname(os.getcwd()), sub_folder_name)  # outside src
    
    validation_passed = True
    print("*** save_train_validate_test_sets() called...")

    data_folder = os.path.join(os.getcwd(), data_folder_name)
    images_folder = os.path.join(data_folder, images_folder_name)
    data_file = os.path.join(data_folder, data_file_name)
    validation_passed = validations_for_create_train_validate_test_sets(data_folder
                                                                        , images_folder, data_file, train_pct, validate_pct)
    if validation_passed and os.path.exists(sub_folder):
        print('Validation Failed: sub_folder_name folder {} already exists.'.format(sub_folder))
        validation_passed = False
    
    if validation_passed:        
        # load data
        X_names, Y = load_labels_and_image_names(images_folder, data_file, limit=limit)

        print('len(Y) =', len(Y), ' Y[:5] = ', Y[:5])
        print('len(X_names) =', len(X_names), ' X_names (first 2, last 2) = ', X_names[:2], X_names[-2:])

        # determine test dataset size (percentage)
        test_pct = np.round(1.0 - (train_pct + validate_pct), 2)
        print('Train, Validate, Test percentages: {}, {}, {}'.format(train_pct, validate_pct, test_pct))

        # Get the shuffled indices for train, validate and test
        train_indices, validate_indices, test_indices = split_indices_proportionate_1s_into_train_validate_test(Y
                                                                                , train_pct, validate_pct, test_pct)

        print('Length of Train, Validate, Test indices: {}, {}, {}'.format(len(train_indices), len(validate_indices), len(test_indices)))

        # create folder to save the files
        os.makedirs(sub_folder, exist_ok=False)
        
        # split labels
        train_Y = Y[train_indices]
        validate_Y = Y[validate_indices]
        test_Y = Y[test_indices]

        # save labels
        data_file_train = os.path.join(sub_folder, 'Y_train.csv')
        data_file_validate = os.path.join(sub_folder, 'Y_validate.csv')
        data_file_test = os.path.join(sub_folder, 'Y_test.csv')
        np.savetxt(data_file_train, train_Y, fmt='%d', delimiter=',', newline='\n')
        np.savetxt(data_file_validate, validate_Y, fmt='%d', delimiter=',', newline='\n')
        np.savetxt(data_file_test, test_Y, fmt='%d', delimiter=',', newline='\n')

        # split image-file-names
        train_X_names = X_names[train_indices]
        validate_X_names = X_names[validate_indices]
        test_X_names = X_names[test_indices]
        
        # save image-file-names
        data_file_train = os.path.join(sub_folder, 'X_names_train.csv')
        data_file_validate = os.path.join(sub_folder, 'X_names_validate.csv')
        data_file_test = os.path.join(sub_folder, 'X_names_test.csv')
        np.savetxt(data_file_train, train_X_names, fmt='%s', delimiter=',', newline='\n')
        np.savetxt(data_file_validate, validate_X_names, fmt='%s', delimiter=',', newline='\n')
        np.savetxt(data_file_test, test_X_names, fmt='%s', delimiter=',', newline='\n')

        # save images in separate folders:
        images_folder_train = os.path.join(sub_folder, 'train')
        images_folder_validate = os.path.join(sub_folder, 'validate')
        images_folder_test = os.path.join(sub_folder, 'test')

        os.makedirs(images_folder_train, exist_ok=False)
        os.makedirs(os.path.join(images_folder_train, '0'), exist_ok=False)
        os.makedirs(os.path.join(images_folder_train, '1'), exist_ok=False)
        
        os.makedirs(images_folder_validate, exist_ok=False)
        os.makedirs(os.path.join(images_folder_validate, '0'), exist_ok=False)
        os.makedirs(os.path.join(images_folder_validate, '1'), exist_ok=False)
        
        os.makedirs(images_folder_test, exist_ok=False)
        os.makedirs(os.path.join(images_folder_test, '0'), exist_ok=False)
        os.makedirs(os.path.join(images_folder_test, '1'), exist_ok=False)
        
        for (fn, label) in zip(train_X_names, train_Y):            
            shutil.copyfile(os.path.join(images_folder, fn), os.path.join(images_folder_train, str(int(label)), fn))

        for (fn, label) in zip(validate_X_names, validate_Y):
            shutil.copyfile(os.path.join(images_folder, fn), os.path.join(images_folder_validate, str(int(label)), fn))

        for (fn, label) in zip(test_X_names, test_Y):
            shutil.copyfile(os.path.join(images_folder, fn), os.path.join(images_folder_test, str(int(label)), fn))
        
        print("... save_train_validate_test_sets() completed ***")


def create_train_validate_test_sets(data_folder_name, images_folder_name, data_file_name, train_pct, validate_pct, limit=0):
    """
    For the provide folder of images and data file name -- prepares the train, validate, test sets
    complete with X : images,  Y: labels, X_names: Additional_info (names of the files)
    """
    data_folder = os.path.join(os.getcwd(), data_folder_name)
    images_folder = os.path.join(data_folder, images_folder_name)
    data_file = os.path.join(data_folder, data_file_name)
    validation_passed = validations_for_create_train_validate_test_sets(data_folder
                                                                        , images_folder, data_file, train_pct, validate_pct)
    if validation_passed:
        # load data
        X_names, Y = load_labels_and_image_names(images_folder, data_file, limit=limit)

        print('len(Y) =', len(Y), ' Y[:5] = ', Y[:5])
        print('len(X_names) =', len(X_names), ' X_names (first 2, last 2) = ', X_names[:2], X_names[-2:])

        # determine test dataset size (percentage)
        test_pct = np.round(1.0 - (train_pct + validate_pct), 2)
        print('Train, Validate, Test percentages: {}, {}, {}'.format(train_pct, validate_pct, test_pct))

        # Get the shuffled indices for train, validate and test
        train_indices, validate_indices, test_indices = split_indices_proportionate_1s_into_train_validate_test(Y
                                                                                , train_pct, validate_pct, test_pct)

        train_Y = Y[train_indices]
        validate_Y = Y[validate_indices]
        test_Y = Y[test_indices]

        train_X_names = X_names[train_indices]
        print("Train images...")
        train_X = load_images(images_folder, train_X_names)
        print('shape', np.shape(train_X), '\n')
        
        validate_X_names = X_names[validate_indices]
        print("Validation images...")
        validate_X = load_images(images_folder, validate_X_names)
        print('shape', np.shape(validate_X), '\n')
        
        test_X_names = X_names[test_indices]
        print("Test images...")
        test_X = load_images(images_folder, test_X_names)
        print('shape', np.shape(test_X), '\n')

        return train_X, train_Y, train_X_names, validate_X, validate_Y, validate_X_names, test_X, test_Y, test_X_names

def normalize_images(train_X, validate_X, test_X):
    """
    divide by 255
    """
    train_processed_X = train_X.astype("float")/255.0
    validate_processed_X = validate_X.astype("float")/255.0
    test_processed_X = test_X.astype("float")/255.0

    return train_processed_X, validate_processed_X, test_processed_X


def add_normalized_day_to_input(train_processed_X, validate_processed_X, test_processed_X):
    """
    Input is images, normalized at this stage. 
    This function adds the Day of the Year (1-365) to the input.
    The value added will also be normalized - i.e. divided by total number of days (365 or 360)
    """
    ### add day of the year (normalized)
    print('Inside add_normalized_day_to_input: number of days =', len(train_processed_X))
    return train_processed_X, validate_processed_X, test_processed_X


def build_CNN_model(input_shape, num_classes, layers_to_add):
    """
    input_shape should be a list of height, width, channels
    layers should be a list of Conv2D layers()
    """
    # Model input
    #input_shape = [image_height, image_width, num_channels] # height, width, channels
    model_input = layers.Input(shape=input_shape)
    
    input_for_layer = model_input  # initial layer's input will be the model_input input layer
    for layerParams in layers_to_add:
        if layerParams['layerType'] == 'C':
            hidden = layers.Conv2D(filters=layerParams['filters'], kernel_size=layerParams['kernel_size'], padding='same', activation='relu')(input_for_layer)
            input_for_layer = hidden  # this layer is input for the next
        if layerParams['layerType'] == 'M':
            hidden = layers.MaxPooling2D(pool_size=layerParams['pool_size'], strides=layerParams['strides'])(input_for_layer)
            input_for_layer = hidden  # this layer is input for the next

    # Evenutally flatten and create dense output layer
    # Flatten
    hidden = layers.Flatten()(hidden)
    # Output Layer
    output = layers.Dense(units=num_classes, activation='sigmoid')(hidden)

    # Create model
    model = Model(model_input, output, name='model_'+str(int(time.time())))

    return model

def create_dataset_from_local_files(ws, datastore_name, files_path, path_on_datastore, dataset_name, dataset_desc, show_progress):
    """"""
    # ref: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-register-datasets
    datastore = Datastore.get(ws, datastore_name)
    print('progressing to upload files...')
    ds = Dataset.File.upload_directory(src_dir=files_path,
            target=DataPath(datastore,  path_on_datastore),
            show_progress=show_progress)    
    print('files uploaded. Now registering the dataset...')    
    ds = ds.register(workspace=ws,
                        name=dataset_name,
                        description=dataset_desc)
    print('dataset_registered!')


def create_tabluar_dataset_from_local_labels_file(ws, datastore_name, files_path, path_on_datastore, dataset_name, dataset_desc, show_progress):
    """"""
    # ref: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-register-datasets
    datastore = Datastore.get(ws, datastore_name)
    print('progressing to upload files...')
    ds = Dataset.File.upload_directory(src_dir=files_path,
            target=DataPath(datastore,  path_on_datastore),
            show_progress=show_progress)    
    print('files uploaded. Now registering the dataset...')    
    ds = ds.register(workspace=ws,
                        name=dataset_name,
                        description=dataset_desc)
    print('dataset_registered!')


def upload_local_images_as_dataset(datastore_name, sub_folder_name):
    """ images from train, test, validate folders with sub-folders by labels, uploaded to a dataset """
    upload_sets = ['train', 'validate', 'test']
    for upds in upload_sets:
        files_path = os.path.join(os.path.dirname(os.getcwd()), sub_folder_name, upds)
        dataset_name = sub_folder_name + '_' + upds
        dataset_desc = sub_folder_name + ' ' + upds + ' images'
        path_on_datastore = dataset_name
        show_progress = True
        create_dataset_from_local_files(ws, datastore_name, files_path, path_on_datastore, dataset_name, dataset_desc, show_progress)


def download_dataset_of_images(dataset_prefix):
    """ images downloaded from a dataset  with train, test, validate folders with sub-folders by labels 
        input: dataset_prefix is the name of the dataset without _train, _test, or _validate.
    """
    if os.path.exists(dataset_prefix):
        print('The folder {} already exists. Will not proceed with the download.'.format(dataset_prefix))        
    else:
        os.makedirs(dataset_prefix, exist_ok=True)
        
        upload_sets = ['train', 'validate', 'test']
        for upds in upload_sets:
            sub_folder_name = os.path.join(dataset_prefix, upds)
            os.makedirs(sub_folder_name, exist_ok=True)
            dataset_name = dataset_prefix + '_' + upds
            dataset = Dataset.get_by_name(ws, dataset_name)  # ws is available in the global variable
            dataset.download(target_path=sub_folder_name, overwrite=False)

        print('Done! Dataset is now available locally.')    


def load_images_subset_and_shuffle(subset_folder, limit=0):
    """
    If limit is not 0, the specified number of images will be considered. Useful for sampling images and testing logic.
    Loading into numpy arrays of images, labels, and image file names
    """
    if not os.path.exists(subset_folder):
        raise ValueError('Error in the call to load_images_subsets(). subset_folder does not exist. Provided parameter', subset_folder)

    images = []
    labels = []
    image_names = []

    for d in os.listdir(subset_folder):
        label_folder = os.path.join(subset_folder, d)
        if os.path.isdir(label_folder):
            print('label folder', label_folder)
            for f in os.listdir(label_folder):
                if f.endswith('.jpg') or f.endswith('.png'):
                    image = cv2.imread(os.path.join(label_folder,f))  # read image
                    images.append(image)
                    image_names.append(f)
                    labels.append(int(d))
                    if limit > 0 and len(image_names) == limit:
                        break
        if limit > 0 and len(image_names) == limit:
            break
    
    # to numpy
    images = np.array(images)
    labels = np.array(labels)
    image_names = np.array(image_names)

    # shuffle        
    if len(image_names) > 1:
        idx = np.random.permutation(len(image_names))
        images, labels, image_names = images[idx], labels[idx], image_names[idx]
    
    return images, labels, image_names

#### Call prep functions -- as required -- 
#### 1) Get Dataset to Local -- to specified data and images folders.
dataset_name = 'images_aoi_107_set_1__1980_2030'
data_folder_name = 'data'
images_folder_name = 'images'
# get_dataset_to_local(dataset_name)  # uncomment to run

#### save to sub-folder -- for area of interest northwest US (107)
train_pct = 0.7
validate_pct = 0.15
data_file_name = '1980_to_2030__hw_area_pct_30_labels.csv'
sub_folder_name = 'aoi_107_2021_11_11'
# save_train_validate_test_sets(data_folder_name, images_folder_name, data_file_name, sub_folder_name, train_pct, validate_pct, limit=0)

#### upload local images as dataset
# Uncomment below when uploading
# upload_local_images_as_dataset('workspaceblobstore', sub_folder_name)

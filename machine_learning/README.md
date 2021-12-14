# heat-extremes-CMIP6 | Machine Learning
Using CMIP data (netCDF) files to identify and analyze extreme heat events like heatwaves and extreme cold days events.

Using the identified heatwave information to generate images for machine learning.

Then using those images for training some machine learning models.

## Repository Contents
1) This repository directory uses the images generated from the code in the other directory named *identify_extreme_temperature*. 
    - In that directory, the notebook named *code_usage_examples_01.ipynb* includes the code for:
        - generating underlying data of heatwaves (extreme temperature), if not already available, for the specified duration.
        - generating images from this data of heatwave yes/no
        - generating labels based on user specified threshold of total area under heatwave for the entire image to be labeled as in heatwave yes, otherwise no.
        - Uploading all the images, as a zip file, to Azure -- as an Azure dataset -- along with the csv files of labels.
    
2) This repository directory, named *machine_learning*, itself has the code for:
    - preparing / managing datasets, including train-validate-test subsets:
        - One time download of images zip file dataset from azure, along with the csv file of labels
        - Using the these images to create subset datasets Train-Validate-Test with proportionate distribution of yes/no labels across all three subsets
        - Uploading back into Azure ML Studio dataset, the three subsets for re-use
        - Downloading the three subset datasets (train-validate-test), on any new environment in ML Studio, where the three subsets are not available 
    - Creating Convolutional Neural Networks with user specified choice of layer details
    - A notebook with the code to create such a model, then train and test using the image subsets

## Usage Notes

### Generate images
As mentioned in repository contents -- the code for generating labeled images for machine learning resides in the directory named *identify_extreme_temperature*.

There, in the notebook *code_usage_examples_01.ipynb*, is the section for 'Generating images for ML', that has the block of code to call the function Generate_ML_Images_and_Labels_By_Region_or_AoI() that:
* Works for the specified duration
    * Years of duration could be something like 2000 to 2050
    * Applicable months could be May, June, July, August, September
* Generates, for that duration, the underlying data for heatwave identification -- if not already available.
* Creates the images for each day of the duration, representing maximum temperature values for the area -- using a fixed color scale for uniform shades of color across all days
* Labels the images as heatwave yes/no (1 or 0), based on the user specified threshold for percentage of area under heatwave. For example, when specifed 0.3 (or 30%), all images with 30% or more of individual grid cells identified with heatwave yes (1) will be labeled as yes (1) for the entire image, else no (0).

### Prepare Environment for Machine Learning

#### Azure ML Studio
We have used Azure ML Studio for our work and found it user-friendly and helpful.

A new ML studio environment is created with some default storage as well. We used the same for storing our data subsets, though any other Azure Storage location could have been used too.

Finally, with minor changes to the code, it can be run on any python notebook environment too where the images and labels are available.

#### Code files
Place the files from this repostiory directory (*machine_learning*), into your development environment -- in the same location.
The .py python script files contain the helpful code that is out of sight from the notebooks. This way, the notebooks have the high level code that the user invokes and interacts with.

For making available the images and labels from our previous effort in the first (available as zip file and csv file), refer to the last part of the python script file hw_training_prep.py. There, uncomment, one-by-one, the statements that do the needful and execute directly the script file.

After all three statements over there, separate subsets of Train, Validate, Test data will be generated with proportionate distribution of yes/no labels and will be available locally.

### Training the model
Refer to the notebook file *hw_training_01.ipynl* for creating the first model, training and testing it using the train-validate-test subsets described above.

One CNN model has been created and trained with 86% test accuracy.

From here on, more models can easily be tried on this setup.

    
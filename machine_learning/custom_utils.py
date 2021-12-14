## Custom util functions for heatwaves related training by team 4, Harvard Extension School
import os
import numpy as np

def delete_all_files_in_the_folder(folder_name, extn = None):
    for filename in os.listdir(folder_name):
        if extn is None or filename.endswith(extn):
            os.remove(filename)

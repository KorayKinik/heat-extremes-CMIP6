#### imports
import io
import os
import gc
# %matplotlib inline
import matplotlib.pyplot as plt

# data management
import xarray as xr
import numpy as np
# import pandas as pd

# Azure Connectivity
import getpass
import azure.storage.blob
from azure.storage.blob import BlobClient

# utilities
import time
import urllib.request
from datetime import datetime
import cftime
import urllib
import csv
import zipfile


#### #########################################################
#### For consistently formatted file names..
#### #########################################################

#### #########################################################
#### Averagesfile name...
#### #########################################################
def get_averages_filename(result_type, cmip, model_name, n_years, start_year, end_year):
    # example: Avg_temp_max_CMIP6__10_yrs__1970_to_1979
    filename = 'Avg_temp_{}_CMIP{}_{}__{}_yrs__{}_to_{}.nc'.format(result_type, cmip, model_name, n_years
                                                                 , start_year, end_year)                
    return filename


#### #########################################################
#### Functions to download files -- from Internet or Azure
#### Functions to upload files -- to Azure
#### #########################################################

#### #########################################################
#### Download from Azure or Internet
#### #########################################################
def download_file(sas_url, filename, overwrite_local_file, from_azure=False, print_msg=False):
    """
    Downloads the specified file from the internet -- or if specified, from Azure blob storage
    If asked to not overwrite, then first checks if the file is available locally and does not download again in that case.
    - sas_url   : url complete with sas token
    - filename  : name of the downloaded file
    - overwrite_local_file : if True, will overwrite, else, if already available locally, will not download again.
    - from_azure: default False. if True, will download from the azure blob storage
    Returns:  True if downloaded, else False
    """
    downloaded = False
    if print_msg:
        print("downloading {}. Is it from azure?: {}".format(filename, from_azure))
#         if not from_azure:
#             print("from url {}".format(sas_url))
    if overwrite_local_file or not (os.path.isfile(filename)):
        if from_azure: 
            blob_client = BlobClient.from_blob_url(sas_url)
            if blob_client.exists():                
                with open(filename, "wb") as my_blob:
                    download_stream = blob_client.download_blob()
                    my_blob.write(download_stream.readall())
                downloaded = True
        else:
            try:
                with open(filename, "wb") as dnld_file:                
                    with urllib.request.urlopen(sas_url) as f:                    
                        dnld_file.write(f.read())
                downloaded = True                    
            except Exception as e:
                print('Exception with url: ', sas_url)
                raise e

    return downloaded

#### #########################################################
#### Save xarray dataset locally, and if asked for, also upload to Azure
#### relies on xarray dataset attribute 'Store as' to use the name to save the file as
#### #########################################################
def SaveResult(results, azure_url_prefix = None, sas_token=None, local_copy=True):
    """
    Create a NetCDF4 file from the xarrary dataset 'results'. Prompts for sas token if pushing to Azure Blob Storage.
    Input parameters:
    - results   :
    - azure_url_prefix : if uploading to blob, this will be the url to the container and the folder 
    - sas_token : if uploading to blob, a sas token with 'write' permissions to the azure blob container
    - local_copy: By default, True, i.e. local_copy will be retained. Setting to False will remove file only 
                  after upload to Azure Blob Storage
    Returns: string. The name of the newly created file.
    """
    # determine the name of the file
    filename = results.attrs['Store as']
    
    # a local copy will initially be saved, in any case
    results.to_netcdf(filename, mode='w', format='NETCDF4')

    # if required to upload to Azure blob storage
    if not azure_url_prefix is None:
        # prepare to time the operation
        start_time = time.time()
        
        # Create a blob client using the local file name as the name for the blob
        sas_url = os.path.join(azure_url_prefix, filename) # add filename to the url prefix
        sas_url = sas_url + "?" + sas_token
        blob_client = BlobClient.from_blob_url(sas_url)

        # Upload the created file
        with open(filename, "rb") as data:
            blob_client.upload_blob(data)

        # if asked to not retain the local copy after use, then delete the file
        if local_copy == False:
            os.remove(filename)
            
        # print out the time it took
        execution_time = (time.time() - start_time)
        print("Complete execution time | SaveResult | (mins) {:0.2f}".format(execution_time/60.0))
    
    return filename



#### #########################################################
#### create SAS url: an Azure url complete with the provided SAS token
#### #########################################################

def create_sas_url(azure_url_prefix, sas_token, filename):
    """
    Creates the URL complete with Azure container, folder, filename + azure security access token (sas). 
    Used in Interactive_GetAverageForRange.    
    Returns: String. sas_url.
    """
    sas_url = os.path.join(azure_url_prefix, filename)  # add filename to the url prefix
    if not (sas_token is None or sas_token == ''):
        sas_url = sas_url + "?" + sas_token
    return sas_url


#### #########################################################
#### upload local files to Azure
#### #########################################################

def upload_file_to_Azure(save_back_in_azure, sas_url, filename, print_msgs=True, overwrite=False):
    """
    Uploads locally available file to Azure -- unless already present there
    prints out statuses.
    Returns: None.
    """
    if save_back_in_azure and sas_url is not None:
        if print_msgs:
            print("{} UTC: Save back in Azure = True. Upload file if not in Azure already".format(datetime.now().strftime("%H:%M:%S")))
        blob_client = BlobClient.from_blob_url(sas_url)
        if overwrite==False and blob_client.exists():
            if print_msgs:
                print("{} UTC: File {} is already in Azure. Not uploading again.".format(datetime.now().strftime("%H:%M:%S"), filename))           
        else:
            # Upload the created file
            if print_msgs:
                print("{} UTC: File {} not in Azure already. Uploading now...".format(datetime.now().strftime("%H:%M:%S"), filename))
            with open(filename, "rb") as data:
                blob_client.upload_blob(data, overwrite=overwrite)
            if print_msgs:
                print("{} UTC: File {} successfully uploaded.".format(datetime.now().strftime("%H:%M:%S"), filename))
                

#### #########################################################
#### check if the specified file is available in Azure
#### #########################################################

def is_file_in_Azure(sas_url):
    """
    Checks if the specified file is in Azure.     
    Returns: Boolean. True if file is available. Else, False.
    """
    if sas_url is None or sas_url == '':
        return False
    
    blob_client = BlobClient.from_blob_url(sas_url)
    return True if blob_client.exists() else False


#### #########################################################
#### remove specified file from local, from Azure
#### helpful when doing some clean up
#### #########################################################

def remove_file(filename, from_local=False, sas_url=None, print_msgs=False):
    """
    If from_local is True, removes file from local
    If sas_url is not None, removes file from Azure, if there.    
    Returns: None
    """
    if from_local:
        if print_msgs:
            print('Remove from local, if available')
        if os.path.exists(filename):
            os.remove(filename)
            if print_msgs:
                print('Found in local, removed')
        else:
            if print_msgs:
                print('Not found in local')
            
    if not (sas_url is None or sas_url == ''):
        if print_msgs:
            print('Remove from Azure, if available')
        blob_client = BlobClient.from_blob_url(sas_url)
        if blob_client.exists():
            blob_client.delete_blob(delete_snapshots='include')
            if print_msgs:
                print('Found in Azure, removed')
        else:
            if print_msgs:
                print('Not found in Azure')


#### ########################################################
#### Load and explore a local netCDF (CMIP) file
#### #########################################################

def get_xarray_dataset_from_netcdf_file(filename):
    return xr.open_dataset(filename)


#### ########################################################
#### Extract a subset of xarray based on specified region boundaries
#### #########################################################

def region_subset(ds_full, top_lat, bottom_lat, left_lon, right_lon):
    """
    For the given xarray dataset, extracts a subset dataset for the specified latitude, longitude boundaries. 
    This helper function also gets used by Interactive_GetAverageForRange()
    Returns: xarray dataset. subset
    """
    full_bottom_lat = ds_full.coords['lat'].values[0]
    full_top_lat = ds_full.coords['lat'].values[-1]
    full_left_lon = ds_full.coords['lon'].values[0]
    full_right_lon = ds_full.coords['lon'].values[-1]
   
    err_msg = ''
    if full_top_lat < top_lat:
        err_msg += ' Full dataset top lat {} is less than subset top lat {}\n'.format(full_top_lat, top_lat)
    if full_bottom_lat > bottom_lat:
        err_msg += ' Full dataset bottom lat {} is more than subset bottom lat {}\n'.format(full_bottom_lat, bottom_lat)
    if full_right_lon < right_lon:
        err_msg += ' full dataset right lon {} is less than subset right lon {}\n'.format(full_right_lon, right_lon)
    if full_left_lon > left_lon:
        err_msg += ' full dataset left lon {} is more than subset left lon {}\n'.format(full_left_lon, left_lon)
    
    if err_msg != '':
        err_msg = 'Error! subset boundaries are beyond the full dataset\n'  + err_msg
        print(err_msg)
        raise ValueError('Error! subset boundaries are beyond the full dataset')
    
    ds_subset = ds_full.sel(lat=list(np.arange(bottom_lat, top_lat+1, 0.25)), lon=list(np.arange(left_lon, right_lon+1, 0.25)), method="nearest")
    return ds_subset


#### ########################################################
#### Extract a subset of xarray based on dates (time)
#### #########################################################

def datewise_subset(ds_full, start_date, end_date):
    """
    For the given xarray dataset, extracts a subset dataset for the specified dates.
    Input dates should be one of these formats: 1) str (mm/dd/yyyy), or 2) datetime, or 3) cftime.DatetimeNoLeap
    Assumes that the dataset uses 'time' dimension.
    Returns: xarray dataset. subset
    """
    time_coord = 'time'
#     for c in list(ds_ext.coords.items()):
#         if c[0] == 'time':
#             time_coord = 'time'
#         if c[0] == 'day':
#             time_coord = 'day'
    
    sdt = None
    edt = None

    try:
        if isinstance(start_date, str):
            sdt = datetime.strptime(start_date, '%m/%d/%Y')
            sdt = cftime.DatetimeNoLeap(sdt.year, sdt.month, sdt.day)
        elif isinstance(start_date, cftime.DatetimeNoLeap):
            sdt = start_date        
        elif isinstance(start_date, datetime):
            sdt = cftime.DatetimeNoLeap(start_date.year, start_date.month, start_date.day)
            
        if isinstance(end_date, str):
            edt = datetime.strptime(end_date, '%m/%d/%Y')        
            edt = cftime.DatetimeNoLeap(edt.year, edt.month, edt.day)
        elif isinstance(end_date, cftime.DatetimeNoLeap):
            sdt = end_date        
        elif isinstance(end_date, datetime):
            end_dt = cftime.DatetimeNoLeap(end_date.year, end_date.month, end_date.day)
    except:
        pass
        
    if sdt is None or edt is None:
        err_msg = 'Error! both start date and end date must be one of these formats:\n 1) str (mm/dd/yyyy), or 2) datetime, or 3) cftime.DatetimeNoLeap.'
        print(err_msg)
        raise ValueError(err_msg)
    
    first_date = ds_full['time'][0].values    # cftime.DatetimeNoLeap
    last_date = ds_full['time'][-1].values     # cftime.DatetimeNoLeap
    
    if (sdt < first_date) or (edt > last_date):
        err_msg = 'Error! Start and End dates {}, {} are outside the \n time range in the dataset: {} to {}.'.format(sdt.strftime('%m/%d/%Y')
                                                                                                                       , edt.strftime('%m/%d/%Y')
                                                                                                                       , str(first_date)[:10]
                                                                                                                       , str(last_date)[:10])
        print(err_msg)
        raise ValueError(err_msg)
    
    ds_subset = ds_full.sel(time=xr.cftime_range(start=sdt, end=edt, calendar="noleap"), method="nearest")
    return ds_subset


#### #################################################################
#### Get results for a given grid cell (latitude, longitude, and day)
#### if also_print is True, finds the variables in the dataset and 
#### prints the name, values
#### #################################################################

def get_results_for_lat_lon(ds, lat, lon, year, month, day, also_print=False):
    """
    Returns subset of the dataset for the specified latitude, longitude and date
    Input:
         - ds     : the xarray dataset 
         - lat    : latitude 
         - lon    : longitude
         - year, month, day : the date for which results need to be shown
         - also_print: if True, print out variable values
    Returns: slice of the dataset
    """
    if year == 0 or month == 0:
        ds_sub = ds.sel(lat = [lat], lon = [lon], day=[day], method="nearest")
    else:
        ds_sub = ds.sel(lat = [lat], lon = [lon], time=[cftime.DatetimeNoLeap(year, month, day)], method="nearest")
    
    if also_print:
        print(ds_sub.coords)
        for var_and_values in list(ds_sub.data_vars.items()):
            print(var_and_values[0])
            print(var_and_values[1].squeeze().values)
            
    return ds_sub

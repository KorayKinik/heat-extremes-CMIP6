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

# plotting - bokeh
import hvplot.xarray
import panel as pn

# cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# custom code
from extreme_temp_utils import *
from regions import Regions
    
#### ####################################################################    
#### 2
#### Code to Generate Average and Record Temperature (minimum and maximum)
#### Relies on NASA NEX generated climate coarse resolution temperature data
#### expects the variable 'tasmax' or 'tasmin'
#### ####################################################################    

#### ####################################################################    
#### 2.1 CreateDatasetForResults()
#### Function to create the empty Results xarray dataset
#### ####################################################################

def CreateDatasetForResults(filename, start_year, end_year, cmip=6, model_name=None):
    """
    A function that loads the CMIP data file into xarray dataset -- for reference,
    then creates and returns a new xarray with matching dimensions and coordinates 
    but with new, empty variables
    
    The variables of the new xarray will be ['tasmaxavg', 'tasmaxrec'] or ['tasminavg', 'tasminrec']
    Similar three dimensions and the coordinates will be there [day: 365 (or 360), lat: 720,  lon: 1440]
    Input Parameter:
    - filename : a CMIP source data file for createing the reference xarray dataset.
    - start_year : should be the integer value for the intended results date range start (e.g. 1950)
    - end_year : should be the integer value for the intended results date range end (e.g. 1959)
    - cmip        : integer number to specify cmip version (5 or 6 or other). Default is 6    
    - model_name  : name of the model, for averages (without _ssp245 at the end). For example: 'GFDL_ESM4'
    Returns: an xarray dataset
    """
    ds_rec = None
    validations_passed = False
    data_type = ''
    
    # validations
    if not (isinstance(start_year, int) and isinstance(end_year, int) and (end_year > start_year)):
        print('Validation Error: start_year and end_year should be integers; end year should be greater than start year.')
    else:
        validations_passed = True
    
    if validations_passed:
        # load the file
        based_on = xr.open_dataset(filename)
        
        var_minmax = list(based_on.data_vars.keys())[0]  # will be tasmax or tasmin
        
        #shape of the dataset
        day_time = based_on.dims.get('time')
        lat = based_on.dims.get('lat')
        lon = based_on.dims.get('lon')        
        
        if var_minmax == 'tasmax':
            data_type = 'Average and record maximum temperature, based on tasmax'
            ds_variables = ['tasmaxavg', 'tasmaxrec']
            ds_data = {'tasmaxavg' : (['day','lat','lon'], np.empty((day_time, lat, lon))), 'tasmaxrec' : (['day','lat','lon'], np.empty((day_time, lat, lon)))}
        else:
            data_type = 'Average and record minimum temperature, based on tasmin'
            ds_variables = ['tasminavg', 'tasminrec']
            ds_data = {'tasminavg' : (['day','lat','lon'], np.empty((day_time, lat, lon))), 'tasminrec' : (['day','lat','lon'], np.empty((day_time, lat, lon)))}

        ds_rec = based_on.copy(deep = True)   
        ds_rec = ds_rec.rename_dims({'time':'day'})
        ds_rec = ds_rec.assign(ds_data)
        ds_rec = ds_rec.drop_vars(var_minmax)
        ds_rec = ds_rec.assign_coords({"day": xr.DataArray(np.arange(1,day_time+1), dims='day')})
        ds_rec = ds_rec.reset_coords('time', drop=True)
        
        n_years = (end_year - start_year) + 1
        
        new_attrs = {'Dataset' : 'Average temperature CMIP{}'.format(cmip),
                     'About dataset' : 'A dataset with average and record temparatures, across years',
                     'Original values' : var_minmax,
                     'Data variables' : ds_variables,
                     'Data description' : data_type,
                     'Range' : str(n_years) + ' years',
                     'Start year' : str(start_year),
                     'End year' : str(end_year),
                     'Years processed' : 0,
                     'Store as' : get_averages_filename(var_minmax[-3:], cmip, model_name, n_years, start_year, end_year)                     
                    }
        
        ds_rec.attrs = new_attrs
    
    return ds_rec
        

#### ####################################################################    
#### 2.2 GenerateAverageOneFileAtATime()
#### An internal function used to update the averages dataset -- 
#### one file at a time.
#### ####################################################################

def GenerateAverageOneFileAtATime(results, lat_indices, lon_indices, start_year, end_year
    , filename):
    """
    Takes one file at a time and updates the average, peak values in the existing 'results' dataset.
    Average and Record temperatures are captured for each cell, i.e. the lat, lon combination in the specified grid.
    The unit of temperature (Kelvin) is retained.
    Input Parameters:
    - results     : xarray dataset with average, record temperature for 365 days
    - lat_indices : applicable index values for latitudes. If None, will calculate for the complete grid
    - lon_indices : applicable index values for longitudes. If None, will calculate for the complete grid.
    - start_year  : start year of the range
    - end_year    : end year of the range
    - filename    : name of the .nc file to process
    Returns:  None. The input xarray dataframe 'results' is updated.
    """
    # load the file
    print(filename)
    xds = xr.open_dataset(filename)
    
    if not (lat_indices is None or lon_indices is None):
        # take applicabel subset of lat and lon
        xds = xds[dict(lat=lat_indices,lon=lon_indices)]
    
    # print("xds", xds)     ## For debugging
    
    # max or min
    var_minmax = list(xds.data_vars.keys())[0]  # will be tasmax or tasmin
        
    # note the variable names in the results dataset
    result_variables = results.attrs['Data variables']

    new_values = xds[var_minmax][:].to_numpy()

    n_years = results.attrs['Years processed']
    
    # if first year in the range
    if n_years == 0:
        results.update({result_variables[0] : (['day','lat','lon'], new_values)       # avg temperature
                        , result_variables[1] : (['day','lat','lon'], new_values)})   # record temperature        
    else:
        if var_minmax == "tasmin":        
            results.update({result_variables[1] : xr.ufuncs.minimum(new_values
                                                                    , results[result_variables[1]])})   # peak min temperature across years
        else:
            results.update({result_variables[1] : xr.ufuncs.maximum(new_values
                                                                    , results[result_variables[1]])})   # peak max temperature across years
        # for both max and min
        results.update({result_variables[0] : np.round((((results[result_variables[0]] * n_years) + new_values) / (n_years+1)),2)})  # avg across years
        results.update({result_variables[1] : np.round(results[result_variables[1]], 2)})  # round off only
        
    
    # in any case...
    n_years += 1
    results.attrs['Years processed'] = n_years



#### ####################################################################    
#### 2.3 PrepareAverageForRange()
#### A function to prepare the average for the desired range of years
#### Assumes that the yearly average files exist locally or in azure
#### ####################################################################

def PrepareAverageForRange(filename_prefix, lat_indices, lon_indices, start_year, end_year, cmip=6
                           , model_name=None, url_prefix=None, sas_token=None, from_azure=False
                           , overwrite_local_file=False, remove_after_use=True):
    """
    Prepares an empty results dataset and then populates in a loop the averages for the specified range of years:
    * downloads the file for each year if url_prefix is set, else no download and the file will be expected to be available locally.      
    * calls the function that processes data one file at time.
    * then deletes the yearly temperature file used in the process
    Note: if url from azure or other location requiring access token, then provide the SAS token as well.
    Average and Record temperatures are captured for each cell, i.e. the lat, lon combination in the specified grid.
    The unit of temperature (Kelvin) is retained.
    Input Parameters:
    - filename_prefix  : initial part of the filename. Year will be appended to it, along with file extension '.nc'
    - lat_indices : applicable index values for latitudes. If None, will calculate for the complete grid
    - lon_indices : applicable index values for longitudes. If None, will calculate for the complete grid.
    - start_year  : start year of the range
    - end_year    : end year of the range
    - cmip        : integer number to specify cmip version (5 or 6 or other). Default is 6
    - model_name  : name of the model, for averages (without _ssp245 at the end). For example: 'GFDL_ESM4'
    - url_prefix  : url to files location, without the filename
                  : for Azure blob storage, specify the url with the container and the folder location
    - sas_token   : A sas token with 'read' permissions to the azure blob container
    - from_azure  : default is False. Set to True if downloading from Azure blob storage
    - overwrite_local_file  : if True, will overwrite, else, if already available locally, will not download again.
    - remove_after_use      : default = True. Set to False to retain the files locally
    Returns:  The xarray dataframe with 'results'
    """
    validations_passed = False
    ds_results = None
    filename = ''
    
    # validations
    if not (isinstance(start_year, int) and isinstance(end_year, int) and (end_year > start_year)):
        print('Validation Error: start_year and end_year should be integers; end year should be greater than start year.')
    else:
        validations_passed = True
    
    if validations_passed:               
        total_years = 0        
            
        # prepare to time the operation
        start_time = time.time()
        for yr in range(start_year, (end_year + 1)):
            print(yr)
            filename = filename_prefix + str(yr) + '.nc'
            
            # if required, download the file so it is available locally
            if not url_prefix is None:
                sas_url = os.path.join(url_prefix, filename) # add filename to the url prefix
                if sas_token is not None:
                    sas_url = sas_url + "?" + sas_token            # add token, if provided
                download_file(sas_url, filename, overwrite_local_file, from_azure)
            
            # first prepare results dataset
            if total_years == 0:
                ds_results = CreateDatasetForResults(filename = filename, start_year = start_year, end_year = end_year
                                                     , cmip=cmip, model_name=model_name)

            # process the file
            GenerateAverageOneFileAtATime(ds_results, lat_indices, lon_indices, start_year, end_year
                                          , filename)
            total_years += 1
            
            # delete the file
            if remove_after_use:
                os.remove(filename)
            
        # print out the time it took
        execution_time = (time.time() - start_time)
        print("Complete execution time | PrepareAverageForRange | (mins) {:0.2f}".format(execution_time/60.0))
            
    return ds_results
    
    
    
#### ####################################################################    
#### 2.4 AggregateAverageForRange()
#### A function to aggregate average results from multiple pre-processed 
####  10 year results into longer ranges
#### ####################################################################

def AggregateAverageForRange(start_year, end_year, cmip=6, model_name=None, result_type='max'
                           , azure_url_prefix=None, sas_token=None
                           , overwrite_local_file=False, remove_after_use=True):
    """
    Makes available Average and Record temperatures for multiple decades. 
    * Date range should be a multiple of decades --- e.g. 1950 to 1969  OR 1951 to 1970.
    * Make sure the underlying data files, by decades are available. Else, an exception will be raised!
    
    Checks the preprocessed average data files in Azure cloud or locally and:
    * downloads the matching file if already exists
    * else, checks for the component files (by decade). 
    * -- If complete data not available, exception is thrown. Else, the required files are downloaded and aggregated.
    * returns the results xarray dataset
    Average and Record temperatures are captured for each cell, i.e. the lat, lon combination in the specified grid.
    The unit of temperature (Kelvin) is retained.
    Input Parameters:
    - start_year  : start year of the range
    - end_year    : end year of the range
    - cmip        : integer number to specify cmip version (5 or 6 or other). Default is 6
    - model_name  : name of the model, for averages (without _ssp245 at the end). For example: 'GFDL_ESM4'
    - result_type : string 'max' or 'min'
    - azure_url_prefix=None : if downloading from Azure blob storage, specify the url with the container and the folder
                              , else specify None to check the files locally.
    - sas_token   : A sas token with 'read' permissions to the azure blob container
    - overwrite_local_file  : if True, will overwrite, else, if already available locally, will not download again.
    - remove_after_use      : default = True. Set to False to retain the files locally
    Returns:  The xarray dataframe with 'results'
    """
    validations_passed = False
    ds_results = None
    filename = ''
    
    n_years = (end_year - start_year) + 1   # range
    
    # validations
    if not (isinstance(start_year, int) and isinstance(end_year, int) and (end_year > start_year)):
        print('Validation Error: start_year and end_year should be integers; end year should be greater than start year.')
    elif not (n_years)%10 == 0:
        print('Validation Error: The total range (end year - start year)+1 should be a multiple of 10.')    
    elif model_name is None:
        print('Validation Error: The model name cannot be None. Should be string value.')    
    else:
        validations_passed = True
    
    if validations_passed:
        n_components = int(n_years / 10)             # number of decades in the range       
        total_years = 0                    
        
        # intended filename
        filename = get_averages_filename(result_type, cmip, model_name, n_years, start_year, end_year)
        
        # intended component filenames
        component_files = [''] * n_components   # as many component files as many decades in the range
        component_available = [False] * n_components   # initialized as none of the files are available
        complete_data = False                   # initialized as file / components are not available
        
        component_start_year = start_year  
        for i in range(n_components):
            component_end_year = component_start_year + 9
            component_files[i] = get_averages_filename(result_type
                                                        ,cmip, model_name,10,component_start_year,component_end_year)
            print('component files[{}] = {}'.format(i, component_files[i]))
            component_start_year += 10

        # prepare to time the operation
        start_time = time.time()
        
        # if local storage is to be checked first
        if azure_url_prefix is None or overwrite_local_file==False:
            if os.path.exists(filename):                # if the complete file exists
                complete_data = True                
            else:
                # check each file exists
                for i in range(n_components):
                    component_available[i] = os.path.exists(component_files[i])
                    
                complete_data = all(component_available)

        # if complete data is not available locally, check if we can look in Azure blob storage
        if not complete_data:
            if azure_url_prefix is not None:
                # check in azure
                # first the complete file...
                sas_url = azure_url_prefix + filename + "?" + sas_token
                blob_client = BlobClient.from_blob_url(sas_url)
                if blob_client.exists():
                    download_file(sas_url, filename, overwrite_local_file, from_azure=True)
                    complete_data = True
                else:
                    for i in range(n_components):
                        # only for files not availble locally, unless overwrite_local_file = True
                        if not component_available[i] or overwrite_local_file:
                            sas_url = azure_url_prefix + component_files[i] + "?" + sas_token
                            blob_client = BlobClient.from_blob_url(sas_url)
                            if blob_client.exists():
                                download_file(sas_url, component_files[i], overwrite_local_file, from_azure=True)
                            else:
                                errMsg = 'Missing component file (on Azure){}'.format(component_available[i])
                                raise ValueError(errMsg)
                    complete_data = True    

                
            else: 
                errMsg = 'Missing component files (locally): {}'.format(', '.join([str(component_files[i]) for i,bv in enumerate(component_available) if bv]))
                raise ValueError(errMsg)


        # complete files should be available locally at this stage
        # start by checking for the complete file first
        if os.path.exists(filename):
            ds_results = xr.open_dataset(filename)  # load the complete file directly
            total_year = n_years
            # delete the file
            if remove_after_use:
                os.remove(filename)
        else:
            # prepare from component files            
            for cfilename in component_files:
                if total_years == 0:    # first file
                    ds_results = xr.open_dataset(cfilename)                    
                else:
                    ds_comp = xr.open_dataset(cfilename)
                    result_variables = ds_results.attrs['Data variables']
                    if remove_after_use:
                        os.remove(cfilename)
                    if result_type == "min":        
                        ds_results.update({result_variables[1] : xr.ufuncs.minimum(ds_comp[result_variables[1]]
                                                                                , ds_results[result_variables[1]])})   # peak min temperature across years
                    else:
                        ds_results.update({result_variables[1] : xr.ufuncs.maximum(ds_comp[result_variables[1]]
                                                                                , ds_results[result_variables[1]])})   # peak max temperature across years
                    # for both max and min
                    ds_results.update({result_variables[0] : (ds_results[result_variables[0]] + ds_comp[result_variables[0]])})    # sum across years
                    
                # in any case    
                total_years += 10
            # finally, for avg from component files        
            ds_results.update({result_variables[0] : np.round((ds_results[result_variables[0]] / n_components), 2)})    # avg across years
            ds_results.update({result_variables[1] : np.round(ds_results[result_variables[1]], 2)})    # round off values

        # set attributes
        ds_results.attrs['Range'] = str(n_years) + ' years'
        ds_results.attrs['Start year'] = str(start_year)
        ds_results.attrs['End year'] = str(end_year)
        ds_results.attrs['Years processed'] = total_years
        ds_results.attrs['Store as'] = get_averages_filename(result_type, cmip, model_name, n_years, start_year, end_year)
            
        # print out the time it took
        execution_time = (time.time() - start_time)
        print("Complete execution time | AggregateAverageForRange | (mins) {:0.2f}".format(execution_time/60.0))
            
    return ds_results

    

#### ####################################################################    
#### Interactive_GetAverageForRange() is the wrapper function for 
#### generating complete range of Averages using all the other functions
#### in this section
#### ####################################################################

#### ####################################################################    
#### 2.5 Validations_for_Interactive_GetAverageForRange()
#### A function to perform validations for the function 
#### Interactive_GetAverageForRange()
#### ####################################################################    

def Validations_for_Interactive_GetAverageForRange(start_year, end_year, n_years, save_back_in_azure, azure_url_prefix, sas_token):
    """
    Performs validations applicable to the function Interactive_GetAverageForRange.    
    Returns: Boolean. True if validations pass. Else returns False.
    """
    validations_passed = False
    if not (isinstance(start_year, int) and isinstance(end_year, int) and (end_year > start_year)):
        print('Validation Error: start_year and end_year should be integers; end year should be greater than start year.')
    elif not (n_years)%10 == 0:
        print('Validation Error: The total range (end year - start year)+1 should be a multiple of 10.')    
    elif (save_back_in_azure == True) and (azure_url_prefix is None or sas_token is None) :
        print('Validation Error: Both azure_url_prefix and sas_token are to be provided if save_back_in_Azure = True.')    
    else:
        validations_passed = True
    
    return validations_passed


#### ####################################################################    
#### 2.6 prepare_component_averages_files()
#### Another functions that gets used by Interactive_GetAverageForRange()
#### helps arrange 10-year average files. first checks locally, then in
#### Azure, else creates those files and optionally uploads to Azure
#### ####################################################################    

def prepare_component_averages_files(save_back_in_azure, azure_url_prefix, sas_token, result_type, cmip, model_name
                                     , start_year, n_years, cmip_files_url, cmip_file_name_prefix, from_azure):
    """
    For the given range, prepares component (10-year) averages files. 
    Checks if available locally, or in Azure. Else prepares it.
    Uploads to azure if save_back_in_azure = True, and not already there.
    This helper function also gets used by Interactive_GetAverageForRange()
    Returns: None
    """
    n_components = int(n_years / 10)
    component_start_year = start_year
    
    print("{} UTC: Arranging for all the component (10-year) files(s)".format(datetime.now().strftime("%H:%M:%S")))
    
    for i in range(n_components):
        component_end_year = component_start_year + 9
        filename = get_averages_filename(result_type,cmip,model_name,10,component_start_year,component_end_year)
        print('prepare_component_averages_files: filename: ', filename)
        if azure_url_prefix is not None: 
            sas_url = create_sas_url(azure_url_prefix, sas_token, filename)
        else:
            sas_url = None
        
        print("{} UTC: Component file {} of {} : --- {} to {} ----------".format(datetime.now().strftime("%H:%M:%S")
                                                                            , i+1, n_components, component_start_year, component_end_year))
        
                
        local = os.path.exists(filename)    # first, check if available locally
        print("{} UTC: Component file available locally? {}".format(datetime.now().strftime("%H:%M:%S"), 'Yes' if local else 'No. Checking in Azure...'))
        
        if local:
            upload_file_to_Azure(save_back_in_azure, sas_url, filename)  # upload, if required            
        else:
            if is_file_in_Azure(sas_url):
                print("{} UTC: Component file available in Azure, downloading...".format(datetime.now().strftime("%H:%M:%S")))
                download_file(sas_url, filename, overwrite_local_file=True, from_azure=True)
                print("{} UTC: Download complete".format(datetime.now().strftime("%H:%M:%S")))
            else:
                print("{} UTC: Component file also not in Azure, preparing...".format(datetime.now().strftime("%H:%M:%S")))
                ds_results = PrepareAverageForRange(filename_prefix = cmip_file_name_prefix, lat_indices = None, lon_indices = None
                                                    , start_year = component_start_year, end_year = component_end_year, cmip= cmip
                                                   , model_name=model_name, url_prefix=cmip_files_url, sas_token=sas_token, from_azure=from_azure
                                                   , overwrite_local_file=False, remove_after_use=True)
                if ds_results is None:
                    raise ValueError('Error! Component file could not be prepared. Check parameters')
                
                if save_back_in_azure:
                    SaveResult(ds_results, azure_url_prefix, sas_token, local_copy=True)
                else:
                    SaveResult(ds_results, azure_url_prefix=None, sas_token=None, local_copy=True)
                
        component_start_year += 10
        print("{} UTC: All the component (10-year) files(s) are now available".format(datetime.now().strftime("%H:%M:%S")))

        
#### ####################################################################    
#### 2.7 plot_avg_and_record_temperatures()
#### Visualize average and record temperatures (entire range)
#### ####################################################################    

def plot_avg_and_record_temperatures(ds_record, latitude, longitude):
    """
    Pass in the dataset and the latitude, longitude.
    Algorithm will find the nearest matching latitude and longitude in the grid.
    Will show plot for the same.
    Returns: None
    """
    fig, ax = plt.subplots()

    fig.set_size_inches(12, 8)

    ds = ds_record.sel(lat=[latitude], lon=[longitude], method="nearest")

    x = ds.get('day')

    pmax = ds['tasmaxrec'].squeeze()
    ax.plot(x, pmax, color='lightgray', alpha=0.9, label='Average Temperature')

    amax = ds['tasmaxavg'].squeeze()
    ax.plot(x, amax, color='gray', alpha=0.6, label='Maximum Temperature')
    
    ax.set_title('Avg and Record Max Temperature - {} to {}'.format(ds_record.attrs['Start year'], ds_record.attrs['End year']))
    ax.set_xlabel('Day')
    ax.set_ylabel('Temperature in Kelvin')
    
    ax.legend()
    plt.show()
    
    
#### ####################################################################    
#### 2.8 Interactive_GetAverageForRange()
#### The key wrapper function that makes available averages for the range
#### checks if the file is available, or generates it.
#### in doing so, checks if component 10-year average files are there
#### or generates those. In doing so, pulls down the temperature files 
#### and removes from the local environment, after calculating the average

#### This function can be called by the user, OR, by downstream code
#### like the one that generates heatwaves data and requires averages
#### ####################################################################    
    
def Interactive_GetAverageForRange(start_year, end_year, cmip=6, result_type='max'
                            , model_name=None, azure_url_prefix=None, sas_token=None
                            , cmip_files_url=None, cmip_file_name_prefix=None, from_azure=False
                            , save_back_in_azure=False, interactive=True):
    """
    Makes available Average and Record temperatures for multiple decades. 
    * Date range should be a multiple of decades --- e.g. 1950 to 1969  OR 1951 to 1970.
    Checks if the preprocessed average data files are already available locally. 
    If not found, checks in Azure cloud. 
    If not found in Azure as well, tries to prepare the output from the available underlying 10 year files, in azure.
    If underlying 10 year files are also not there then tries to create those as well.
    In that case, checks for underlying source CMIP data files based on the url provided.
    All the files not found in Azure are attempted to be saved there, if azure_url_prefix is provided 
    and save_back_in_azure = True.
    
    * returns the results xarray dataset
    Average and Record temperatures are captured for each cell, i.e. the lat, lon combination in the specified grid.
    The unit of temperature (Kelvin) is retained.
    Input Parameters:
    - start_year  : start year of the range
    - end_year    : end year of the range
    - cmip        : integer number to specify cmip version (5 or 6 or other). Default is 6
    - result_type : string 'max' or 'min'
    - model_name  : Specify the name of the model, for example 'GFDL-ESM4'. This will become a part of the filename.
    - azure_url_prefix=None : if downloading from Azure blob storage, specify the url with the container and the folder
                              , else specify None to check the files locally.
    - sas_token   : A sas token with 'read' permissions to the azure blob container    
    - cmip_files_url: Underlying cmip files could be at a different path in azure, or elsewhere.
    - cmip_file_name_prefix: Name of the underlying cmip files, without the year. E.g. 'tasmin_day_BCSD_historical_r1i1p1_inmcm4_'
    - from_azure  : True, if the cmip files path is of azure. In that case sas_token will be used to read/write files.
    - save_back_in_azure: If true, the final file
    - interactive: If true, will prompt for user confirmation before executing. Set False to bypass user input.
    Returns:  The xarray dataframe with 'results'
    """
    validations_passed = False
    ds_results = None
    filename = ''
    sas_url = None
    
    n_years = (end_year - start_year) + 1   # range
    
    # proceed if validations pass
    validations_passed = Validations_for_Interactive_GetAverageForRange(start_year, end_year, n_years
                                                                        , save_back_in_azure, azure_url_prefix, sas_token)
    
    if validations_passed:
        n_components = int(n_years / 10)             # number of decades in the range       
        total_years = 0   
        
        print('Get Average and Record {} temperature for the range {} to {}, for CMIP{}, model {}'.format(result_type, start_year
                                                                                                          , end_year, cmip, model_name))
        
        if interactive:
            ans = input('Do you want to continue? y/n').lower()
        else:
            ans = 'y'
        
        if not ans == 'y':
            print("Response: {}.\n Not 'y', stopping execution.".format(ans))        
            return None
        
        start_time = time.time()        # prepare to time the operation
        if model_name.endswith('_ssp245') or model_name.endswith('_ssp585'):
            model_name = model_name[:-7]
                
        filename = get_averages_filename(result_type, cmip, model_name, n_years, start_year, end_year)
        print("{} UTC: Get Averages file {}".format(datetime.now().strftime("%H:%M:%S"), filename))
        
        if azure_url_prefix is not None: 
            sas_url = create_sas_url(azure_url_prefix, sas_token, filename)
        
        local = os.path.exists(filename)    # first, check if available locally
        print("{} UTC: Available locally? {}".format(datetime.now().strftime("%H:%M:%S"), 'Yes' if local else 'No. Checking in Azure...'))
        
        if local:
            upload_file_to_Azure(save_back_in_azure, sas_url, filename)  # upload, if required
            ds_results = xr.open_dataset(filename)                                # open file        
        else:
            if is_file_in_Azure(sas_url):
                print("{} UTC: Available in Azure {}, downloading...".format(datetime.now().strftime("%H:%M:%S"), filename))
                download_file(sas_url, filename, overwrite_local_file=True, from_azure=True)
                print("{} UTC: Download complete".format(datetime.now().strftime("%H:%M:%S")))
                ds_results = xr.open_dataset(filename)
            else:
                if sas_url is None or sas_url == '':
                    print("{} UTC: Azure url not provided, could not check there. Now preparing...".format(datetime.now().strftime("%H:%M:%S")))
                else:
                    print("{} UTC: Also not in Azure {}, preparing...".format(datetime.now().strftime("%H:%M:%S"), filename))
                prepare_component_averages_files(save_back_in_azure, azure_url_prefix, sas_token, result_type, cmip, model_name
                                                 , start_year, n_years, cmip_files_url, cmip_file_name_prefix, from_azure)
            
                print("{} UTC: Aggregate 10-year files into full range...".format(datetime.now().strftime("%H:%M:%S")))
                ds_results = AggregateAverageForRange(start_year, end_year, cmip, model_name, result_type
                           , azure_url_prefix, sas_token
                           , overwrite_local_file=False, remove_after_use=False)
                
                if save_back_in_azure:
                    print("{} UTC: Dataset ready, saving locally and in Azure...".format(datetime.now().strftime("%H:%M:%S"), filename))                
                    SaveResult(ds_results, azure_url_prefix, sas_token, local_copy=True)
                else:
                    print("{} UTC: Dataset ready, saving locally...".format(datetime.now().strftime("%H:%M:%S"), filename))                
                    SaveResult(ds_results, azure_url_prefix=None, sas_token=None, local_copy=True)
        
         # print out the time it took
        execution_time = (time.time() - start_time)
        print("Complete execution time | Interactive_GetAverageForRange | (mins) {:0.2f}".format(execution_time/60.0))
        
    return ds_results


#### ####################################################################    
#### 3
#### Code to Generate Continous Days Extreme Temperature (minimum and maximum)
#### That is, cold spells and heat waves
#### expects the variable 'tasmax' or 'tasmin'
#### ####################################################################    

#### ####################################################################    
#### 3.1 Validations_for_Identify_1_year_Extreme_Temp_By_Region()
#### Helper function to perform validations
#### ####################################################################

def Validations_for_Identify_1_year_Extreme_Temp_By_Region(analysis_year, threshold
                                                , n_continuous_days, region_id, area_of_interest, name_of_area_of_interest
                                                , based_on_averages, averages_start_year, averages_end_year):
    """
    Performs validations applicable to the function Identify_1_year_Extreme_Temp_By_Region.    
    Returns: Boolean. True if validations pass. Else returns False.
    """
    validations_passed = False
    regions = Regions()
       
    if not isinstance(analysis_year, int):
        print('Validation Error: analysis_year must be a valid integer')
    elif regions.get_region_by_ID(region_id) is None:
        print('Validation Error: Invalid region ID. Valid Regions are:')
        for r in regions.get_all_regions().items():
            print(' ', r[0], '-', r[1]['region_name'])
    elif not (isinstance(threshold, int) or isinstance(threshold, float)):
        print('Validation Error: threshold must be a numeric value.')    
    elif not isinstance(n_continuous_days, int):
        print('Validation Error: n_continuous_days must be an integer value.')
    elif area_of_interest is not None and not isinstance(area_of_interest, dict):
        print("Validation Error: Area of interest, if provided, must be a dict with the keys 'top_lat', 'bottom_lat', 'left_lon', 'right_lon'.")        
    elif area_of_interest is not None and ('top_lat' not in area_of_interest.keys() \
                                               or 'bottom_lat' not in area_of_interest.keys() \
                                               or 'left_lon' not in area_of_interest.keys() \
                                               or 'right_lon' not in area_of_interest.keys()):
        print("Validation Error: Area of interest, if provided, must be a dict with the keys 'top_lat', 'bottom_lat', 'left_lon', 'right_lon'.")        
    elif area_of_interest is not None and (name_of_area_of_interest is None or name_of_area_of_interest==''):
        print("Validation Error: Area of interest, if provided, then a Name for the area of interest must also be provided.")        
    elif based_on_averages and not (isinstance(averages_start_year, int) and isinstance(averages_end_year, int)):
        print("Validation Error: If based_on_averages = True, then provide integer values for averages_start_year and averages_end_year.")        
    else:
        validations_passed = True
    
    return validations_passed


#### ####################################################################    
#### 3.2 get_1_year_extreme_temp_filename()
#### Helper function to generate consistently formatted file names.
#### ####################################################################

def get_1_year_extreme_temp_filename(result_type, cmip, model_name
                               , region_id, analysis_year, name_of_area_of_interest
                               , based_on_averages, threshold, is_percentage, n_continuous_days
                               , averages_start_year, averages_end_year):
    # example: Ext_max_t__Rgn_1__2025__Abv_5_K_for_3_days__CMIP6_ssp245_Avg_yrs_1990_09
    
    if based_on_averages:
        criteria = '{}_Avg_{}_{}'.format('Abv' if result_type == 'max' else 'Blw', threshold, 'pct' if is_percentage else 'K')
        avgs_year = '_Avg_yrs_{}_{}'.format(averages_start_year, str(averages_end_year)[-2:])
    else:
        criteria = '{}_{}_K'.format('Abv' if result_type == 'max' else 'Blw', threshold)
        avgs_year = ''
                                           
    criteria += '_for_{}_days'.format(n_continuous_days)
    
    
    filename = 'Ext_{}_t__Rgn_{}{}__{}__{}__CMIP{}_{}{}.nc'.format(result_type, region_id
                                                                         , '' if name_of_area_of_interest is None else '_AoI_'+ str(name_of_area_of_interest)
                                                                         , analysis_year, criteria, cmip, model_name, avgs_year)                
    return filename


#### ####################################################################    
#### 3.3 window_gte_threshold()
#### Helper function to determine for rolling windows, if above threshold
#### of 0, or not.
#### ####################################################################

def window_gte_threshold(ds, axis):
    """
    Determines for each rolling window if All the days in the window are equal to or above the threshold
    Returns: a multi-dimensional array of the same size as the input
    """
      
    return np.all(np.greater_equal(ds, 0), axis=-1)


#### ####################################################################    
#### 3.4 Identify_1_year_Extreme_Temp_By_Region()
#### Function to create the xarray dataset and the .nc file for 
#### 1 year data of days with extreme temperature (cold spells or heat waves)
#### ####################################################################

def Identify_1_year_Extreme_Temp_By_Region(analysis_year, threshold, result_type = 'max'
                            , based_on_averages=False, is_percentage=False
                            , n_continuous_days=3, cmip=6, model_name='historical', region_id=1
                            , averages_start_year=None, averages_end_year=None
                            , azure_url_extremes=None, sas_token=None
                            , azure_url_averages=None, azure_url_1_year_temp=None, url_1_year_temp=None, temp_filename=None
                            , remove_source_files=True, area_of_interest=None, name_of_area_of_interest=None
                            , interactive=False, print_extra_msg=False):
    """
    Makes available the extreme maxiumn temperature identification data, for analysis of 1-year duration,
     for the specified region and optionally, for the specified area of interest
    If azure_url_extremes is provided then: 
      the file is first looked up in Azure and downloaded.
      else, the file is generated and then uploaded back in to Azure.
    Else - the file is generated.
    Assumes that the source data files (averages and 1-yr temp) are already available locally, or in Azure, or elsewhere on Internet. 
      Please generate those separately if not already available.    
    
    The unit of temperature (Kelvin) is retained.
    Input Parameters:
    - analysis_year        : The year for which .
    - threshold            : a fixed temperature, or a value or pecentile above average that will be considered for extreme temp
    - result_type          : 'max' or 'min' for analysis of extreme Maximum or Minimum temperatures
    - based_on_averages    : if True, the threshold is the difference above the average temperature. Else, actual temp in Kelvin
    - is_percentage        : True, a percentile above average. False, a fixed value above average. Evaluated only if based_on_averages=True. 
    - n_continuous_days    : number of continuous days of above threshold temperature to qualify as a extreme temp event.
    - cmip                 : integer number to specify cmip version (5 or 6 or other). Default is 6
    - model_name           : Specify the name of the model, for example 'ssp245'. This will become a part of the filename.
    - region_id            : integer ID of the region. See Regions class for available regions.
    - averages_start_year, averages_end_year: range of years to use for average temperatures
    - azure_url_extremes   : if provided, the extreme temperature identification results will be uploaded in Azure for reuse.
    - sas_token            : A sas token with 'read' permissions to the azure blob container. If saving back, then 'write' permission is required.  
    - azure_url_averages   : url without filename. if averages file is to be downloded from azure. will be used only if not found locally.
    - azure_url_1_yr_temp  : url without filename. if 1-year temperature files are to be downloded from azure. will be used only if not found locally.
    - url_1_yr_temp        : url without filename. if 1-year temperature files are to be downloded from the Internet. will be used only if not found locally.
    - temp_data_filename   : the name of the 1-year temperature file must be provided
    - remove_source_files  : If downloaded, remove from local environment, after use, the source files (averages file and 1-year temperature files)
    - area_of_interest     : if provided, must be a dict with the keys: 'top_lat', 'bottom_lat', 'left_lon', 'right_lon'
    - name_of_area_of_interest : string value if also provided area_of_interest, else None. A name to distinguish the results file.
    - print_extra_msg      : default is False. If True, will print out extra information during the processing.
    - interactive : If true, will prompt for user confirmation before executing. Set False to bypass user input.
    Returns: tuple of 1) xarray of the 1-year extremes data file in the range and 2) xarray of the averages, if used, else None.
    """
    validations_passed = False
    ds_results = None
    ds_averages = None
    filename = None
    averages_filename = None
    temp_1_year_filename = None
    sas_url = None
    
    top_lat = None
    bottom_lat = None
    left_lon = None
    right_lon = None
    
    downloaded_avgs_file = False
    downloaded_temp_file = False
    
    validations_passed = Validations_for_Identify_1_year_Extreme_Temp_By_Region(analysis_year, threshold
                                                            , n_continuous_days, region_id, area_of_interest, name_of_area_of_interest
                                                            , based_on_averages, averages_start_year, averages_end_year)
    
    reg = Regions().get_region_by_ID(region_id)
    
    if validations_passed:
        print_msg = '\nGet Extreme maximum temperature identification data\n ... for the year {},'.format(analysis_year)
        print_msg += '\n ... for the region {}-{}'.format(region_id, reg['region_name'])
        print_msg += '\n ... for model {} of CMIP{}'.format(model_name, cmip)
        if area_of_interest is not None:
            print_msg += '\n ... area of interest: top_lat{}, bottom_lat{}, left_lon, right_lon'.format(area_of_interest.get('top_lat'),
                                                                                                    area_of_interest.get('bottom_lat'),
                                                                                                    area_of_interest.get('left_lon'),
                                                                                                    area_of_interest.get('right_lon'))
        if based_on_averages:
            print_msg += '\n ... for {}{} above average from the years {} to {}'.format(threshold, '%' if is_percentage else ' Kelvin'
                                                                           , averages_start_year, averages_end_year)
        else:
            print_msg += '\n ... for temperature above {} Kelvin'.format(threshold)
        print_msg += '\n ... when observed for {} continuous days\n'.format(n_continuous_days)
        
        print(print_msg)
        
        if interactive:
            ans = input('Do you want to continue? y/n').lower()
        else:
            ans = 'y'
        
        if not ans == 'y':
            print("Response: {}.\n Not 'y', stopping execution.".format(ans))        
            return None
        
        start_time = time.time()        # prepare to time the operation
        
        # file naming patterns        
        # Extreme_max_t__Rgn_1__2025__Abv_Avg_3_K_for_3_days__CMIP6_ssp245.nc
        # Extreme_max_t__Rgn_1_AoI_US__2025__Abv_Avg_2_pct_for_4_days__CMIP6_ssp245.nc
        # Extreme_min_t__Rgn_1_AoI_US__2025__Blw_270_K_for_4_days__CMIP6_ssp245.nc
        
        if based_on_averages:
            criteria = 'Abv_Avg_{}_{}'.format(threshold, 'pct' if is_percentage else 'K')
        else:
            criteria = 'Abv_{}_K'.format(threshold)
        criteria += '_for_{}_days'.format(n_continuous_days)
        
        extremes_filename = get_1_year_extreme_temp_filename(result_type, cmip, model_name
                               , region_id, analysis_year, name_of_area_of_interest
                               , based_on_averages, threshold, is_percentage, n_continuous_days
                               , averages_start_year, averages_end_year)

        print('{} UTC: Extreme temperature data filename: {}'.format(datetime.now().strftime("%H:%M:%S"), extremes_filename))        
        filename = extremes_filename
        downloaded_from_azure = False
        local = os.path.exists(filename)    # first, check if available locally
        print("{} UTC: Already available locally? {}".format(datetime.now().strftime("%H:%M:%S"), 'Yes' if local else 'No. Checking in Azure...'))
        if not local:            
            if azure_url_extremes is not None: 
                sas_url = create_sas_url(azure_url_extremes, sas_token, filename)
                if is_file_in_Azure(sas_url):
                    print("{} UTC: Available in Azure, downloading...".format(datetime.now().strftime("%H:%M:%S")))
                    download_file(sas_url, filename, overwrite_local_file=True, from_azure=True)
                    downloaded_from_azure = True
                    print("{} UTC: Download complete".format(datetime.now().strftime("%H:%M:%S")))                    
                else:
                    print("{} UTC: Also not in Azure. Preparing the file...".format(datetime.now().strftime("%H:%M:%S")))
                    
            else:
                print("{} UTC: URL not provided to check in Azure. Preparing the Extreme temp data file... ".format(datetime.now().strftime("%H:%M:%S")))
        
        if not local and not downloaded_from_azure:   # not found in azure as well, prepare file
           # first, get 1 year temp file
            filename = temp_filename               
            print('{} UTC: Working with 1_year temperature filename: {}'.format(datetime.now().strftime("%H:%M:%S"), filename))            
            local = os.path.exists(filename)    # first, check if available locally
            print("{} UTC: Available locally? {}".format(datetime.now().strftime("%H:%M:%S"), 'Yes' if local else 'No. Checking in Azure / Internet...'))
            if not local:            
                if azure_url_1_year_temp is not None: 
                    sas_url = create_sas_url(azure_url_1_year_temp, sas_token, filename)
                    if is_file_in_Azure(sas_url):
                        print("{} UTC: Available in Azure {}, downloading...".format(datetime.now().strftime("%H:%M:%S"), filename))
                        download_file(sas_url, filename, overwrite_local_file=True, from_azure=True)
                        print("{} UTC: Download complete".format(datetime.now().strftime("%H:%M:%S")))                    
                    else:
                        print("{} UTC: Also not in Azure {}. Error!".format(datetime.now().strftime("%H:%M:%S"), filename))
                        raise ValueError('Required file {} not found in the local environment or in Azure.'.format(filename))
                elif url_1_year_temp is not None: 
                    download_file(os.path.join(url_1_year_temp, filename), filename, overwrite_local_file=True, from_azure=False, print_msg=False)
                else:
                    raise ValueError('Required file not found in the local environment. Url not provided to check in Azure or Internet.')

                downloaded_temp_file = True  # If not local, and here, means the tempfile was downloaded
            
            ds_1_year_temp = xr.open_dataset(filename)

            # Determine applicable area of interest 
            if area_of_interest is not None:
                top_lat = area_of_interest.get('top_lat')
                bottom_lat = area_of_interest.get('bottom_lat')
                left_lon = area_of_interest.get('left_lon')
                right_lon = area_of_interest.get('right_lon')                
            else:
                top_lat = reg['top_lat']
                bottom_lat = reg['bottom_lat']
                left_lon = reg['left_lon']
                right_lon = reg['right_lon']

            # take subset of full grid temp file
            ds_1_year_temp = region_subset(ds_1_year_temp, top_lat, bottom_lat, left_lon, right_lon)

            if based_on_averages:
                # get averages file
                n_years = (averages_end_year - averages_start_year) + 1                
                if model_name.endswith('_ssp245') or model_name.endswith('_ssp585'):
                    avgs_model_name = model_name[:-7]
                else:
                    avgs_model_name = model_name
                averages_filename = get_averages_filename(result_type, cmip, avgs_model_name, n_years, averages_start_year, averages_end_year)                
                print('{} UTC: Working with Averages filename: {}'.format(datetime.now().strftime("%H:%M:%S"), averages_filename))
                filename = averages_filename
                local = os.path.exists(filename)    # first, check if available locally
                print("{} UTC: Available locally? {}".format(datetime.now().strftime("%H:%M:%S"), 'Yes' if local else 'No. Checking in Azure...'))
                if not local:            
                    if azure_url_1_year_temp is not None: 
                        sas_url = create_sas_url(azure_url_1_year_temp, sas_token, filename)
                        if is_file_in_Azure(sas_url):
                            print("{} UTC: Available in Azure {}, downloading...".format(datetime.now().strftime("%H:%M:%S"), filename))
                            download_file(sas_url, filename, overwrite_local_file=True, from_azure=True)
                            print("{} UTC: Download complete".format(datetime.now().strftime("%H:%M:%S")))                    
                        else:
                            print("{} UTC: Also not in Azure {}. Error!".format(datetime.now().strftime("%H:%M:%S"), filename))
                            raise ValueError('Required file {} not found in the local environment or in Azure.'.format(filename))
                    else:
                        raise ValueError('Required file {} not found in the local environment. Url not provided to check in Azure.'.format(filename))
                    
                    downloaded_avgs_file = True  # If not local, and here, means the averages file was downloaded

                print("{} UTC: Getting subset of applicable region, apply threshold...".format(datetime.now().strftime("%H:%M:%S")))
                ds_averages = xr.open_dataset(filename)
                # subset of averages
                ds_averages = region_subset(ds_averages, top_lat, bottom_lat, left_lon, right_lon)

                ds_1_year_temp = ds_1_year_temp.assign(above_threshold=lambda x: x.tasmax + 0)   # first create the new column with the original temperature
                if result_type == 'max':
                    if is_percentage:
                        ds_1_year_temp.above_threshold[:,:,:] = \
                            ds_1_year_temp.above_threshold[:,:,:] - (ds_averages.tasmaxavg.values * (1 + (threshold/100)))
                    else:
                        ds_1_year_temp.above_threshold[:,:,:] = \
                            ds_1_year_temp.above_threshold[:,:,:] - (ds_averages.tasmaxavg.values + threshold)
                else:
                    raise ValueError('min --- not fully implemented as yet')
                    
            else:       # based_on_averages=False
                if result_type == 'max':
                    ds_1_year_temp = ds_1_year_temp.assign(above_threshold=lambda x: x.tasmax - threshold)
                else:
                    raise ValueError('min --- not fully implemented as yet')
                    
                
                ds_averages = None

            print("{} UTC: Preparing extreme temperature data...".format(datetime.now().strftime("%H:%M:%S")))

            # determine extreme Y/N, using rolling windows of n_continuous_days
            rolling = ds_1_year_temp.above_threshold.rolling(time=n_continuous_days)
            arr_win_status = window_gte_threshold(rolling.construct("window_dim").values,0)
            arr_hw = np.zeros(np.shape(ds_1_year_temp.tasmax))
            n_idx = n_continuous_days - 1
            for i in range(n_idx):
                arr_hw[i:i-n_idx,:,:] =   xr.ufuncs.logical_or(arr_hw[i:i-n_idx,:,:], arr_win_status[n_idx:,:,:])
            arr_hw[n_idx:,:,:] =   xr.ufuncs.logical_or(arr_hw[n_idx:,:,:], arr_win_status[n_idx:,:,:])

            ds_1_year_temp = ds_1_year_temp.assign(extreme_yn = (ds_1_year_temp.dims, arr_hw))

            ds_results = ds_1_year_temp

            # update attributes 
            ds_results.attrs['Data description'] = 'maximum temperature; difference from theshold; 1 or 0 for extreme yes or no'
            ds_results.attrs['Years processed'] = 1
            ds_results.attrs['Store as'] = extremes_filename
            
            new_attrs = {'Dataset' : '1-year Extreme {} Temperature Data CMIP{} {} region_id: {}'.format(result_type, cmip, model_name, region_id),
                     'About dataset' : '1-year Extreme {} Temp Data, for CMIP{} model: {}, for the region {}-{}'.format(result_type
                                                                                                                            , cmip, model_name
                                                                                                                            , region_id
                                                                                                                        , reg.get('region_name')),
                     'Data variables' : 'tas{}, {}, extreme_yn'.format(result_type, 'above_threshold' if result_type=='max' else 'below_threshold'),
                     'Data description' : '{} temperature; difference from threshold; extreme (y/n)-continuous for specified day'.format(result_type),
                     'Range' : '1 year',
                     'Analysis year' : str(analysis_year),
                     'threshold' : region_id,
                     'based_on_averages' : str(based_on_averages),
                     'averages_start_year' : str(averages_start_year),
                     'averages_end_year' : str(averages_end_year),
                     'threshold' : threshold, 
                     'result_type' : result_type,
                     'is_percentage' : str(is_percentage),
                     'Number of continuous days to be considered extreme' : n_continuous_days,
                     'cmip' : cmip, 
                     'model_name': model_name,                         
                     'region_id' : region_id,
                     'region_name' : reg.get('region_name'),
                     'region_top_lat' : reg.get('top_lat'),
                     'region_bottom_lat' : reg.get('bottom_lat'),
                     'region_left_lon' : reg.get('left_lon'),
                     'region_right_lon' : reg.get('right_lon'),
                     'region_img_url' : reg.get('img_url'),
                     'Years processed' : 1,
                     'Store as': extremes_filename
                    }
        
            ds_results.attrs = new_attrs


            # save, and upload to azure - if azure_url_extremes is provided AND not already downloaded from there
            if downloaded_from_azure:
                SaveResult(ds_results, azure_url_prefix = None, sas_token=None, local_copy=True)  # save only
            else:
                SaveResult(ds_results, azure_url_prefix = azure_url_extremes, sas_token=sas_token, local_copy=True) # attempt to upload

            print('{} UTC: Saved locally: Extreme temp data file: {}'.format(datetime.now().strftime("%H:%M:%S"), extremes_filename))
            if azure_url_extremes is not None:
                print('{} UTC: Also uploaded to Azure storage'.format(datetime.now().strftime("%H:%M:%S"), extremes_filename))

            if remove_source_files:
                if downloaded_avgs_file:
                    os.remove(averages_filename)
                    print('{} UTC: Averages file was downloaded, now removed - as requested'.format(datetime.now().strftime("%H:%M:%S")))
                    
                if downloaded_temp_file:
                    os.remove(temp_filename)
                    print('{} UTC: 1-year temperature file was downloaded, now removed - as requested'.format(datetime.now().strftime("%H:%M:%S")))
                
        else:   # in local or downloaded_from_azure
            ds_results = xr.open_dataset(filename)
            if azure_url_extremes is not None:
                sas_url = os.path.join(azure_url_extremes, filename)
                if sas_token is not None:
                    sas_url += '?' + sas_token
                upload_file_to_Azure(save_back_in_azure=True, sas_url=sas_url, filename=filename)
            
        # print out the time it took
        execution_time = (time.time() - start_time)
        print("Complete execution time | Identify_1_year_Extreme_Temp_By_Region | (mins) {:0.2f}".format(execution_time/60.0))
        
    return ds_results, ds_averages

    

#### ####################################################################    
#### 3.5 verify_ext_data()
#### A function to verify the extreme temperature data
#### ####################################################################

def verify_ext_data(ext_filename, ds_avg, day_start_idx, day_end_idx, test_lat, test_lon):
    ds_ext = xr.open_dataset(ext_filename)

    result_type = ds_ext.attrs['result_type']
    temp_var = 'tasmax' if result_type=='max' else 'tasmin'
    avg_var = temp_var + 'avg'
    threshold_var = 'above_threshold' if result_type=='max' else 'below_threshold'
    
    # specify range of days to test:
    days = list(range(day_start_idx, day_end_idx+1))

    print('days: {}'.format(days))
    print('lat: {}, lon: {}\n'.format(test_lat, test_lon))

    avg_for_30_yrs_from_file = ds_avg[avg_var].sel(lat=[test_lat], lon=[test_lon], method="nearest")[days].squeeze().values
    print(avg_for_30_yrs_from_file, ': average for this day, from 30-year avg file')

    threshold = int(ds_ext.attrs['threshold'])
    print(threshold, ': threshold')

    print(avg_for_30_yrs_from_file + threshold, ': temp value that will be considered {} threshold.\n'.format(threshold_var[:5]))

    print(np.round(ds_ext[temp_var].sel(lat=[test_lat], lon=[test_lon], method="nearest").values[days].squeeze(), 2), temp_var)
    print(np.round(ds_ext[threshold_var].sel(lat=[test_lat], lon=[test_lon], method="nearest").values[days].squeeze(), 2), threshold_var)
    print(ds_ext.extreme_yn.sel(lat=[test_lat], lon=[test_lon], method="nearest").values[days].squeeze(), 'extreme yn')
    
    print('\n--------------------\ntotal number of days of extreme {} temperature'.format(result_type))
    n_ext_yr = np.sum(ds_ext.extreme_yn.sel(lat=[test_lat], lon=[test_lon], method="nearest").values.squeeze())
    print(n_ext_yr, ' = {}% of days'.format(round((n_ext_yr/365)*100, 1)))
    
    print('\n--------------------\ntotal number of grid cells of extreme {} temperature, one day'.format(result_type))
    n_day = 180
    nlat = int(ds_ext.coords.get('lat').count().values)
    nlon = int(ds_ext.coords.get('lon').count().values)
    print('Total grid cells (lat x lon) = ({} x {}) ='.format(nlat, nlon), (nlat * nlon))
    n_ext_cells = int(np.sum(ds_ext.extreme_yn.values[n_day].squeeze()))
    print(n_ext_cells, '= {}% of the area, for one day, index: {}'.format(round((n_ext_cells/(nlat * nlon))*100, 1), n_day))
    
    return ds_ext



#### ####################################################################    
#### 3.6 interactive_visual_pct_pixels_extreme_by_day()
#### A function to analyze Heatwaves, extreme cold Temperature 
#### from 1-year data results -- 
#### Visualize the Percentage of pixels in extreme (heatwave or cold)
#### Returns - interactive bokeh pane
#### ####################################################################

def interactive_visual_pct_pixels_extreme_by_day(ds_ext):
    """
    Returns: 1) xarray DataArray, with the number of pixels with extreme y/n flag as 1 (True), 
     for each day in the dataset
           : 2) bokeh pane for interactive visualization.
    """
    result_type = ds_ext.attrs['result_type']
    temp_var = 'tasmax' if result_type=='max' else 'tasmin'
    day_wise_mean = ds_ext.extreme_yn.sum(dim=['lat','lon']) / ds_ext[temp_var].count(dim=['lat','lon'])
    dt_index = day_wise_mean.indexes['time'].to_datetimeindex()
    day_wise_mean = xr.DataArray(day_wise_mean, coords={'time':dt_index.values})

    day_wise_mean_plot = day_wise_mean.hvplot()

    pane = pn.panel(day_wise_mean_plot)
    return pane


#### ####################################################################    
#### 3.7 interactive_visual_region_in_polar()
#### A function to analyze Heatwaves, extreme cold Temperature 
#### from 1-year data results
#### Visualize the temperature in polar coordinates (heatwave or cold)
#### Returns - interactive bokeh pane
#### ####################################################################

def interactive_visual_region_in_polar(ds_ext, start_day=1, end_day=5, step_n_days=1, data_variable_name='tasmax'):
    """"""
#     proj = ccrs.Orthographic(-90, 30)
#     proj = ccrs.OSNI(ccrs.TransverseMercator(90,30,0,0))
    proj = ccrs.PlateCarree(central_longitude=0.0, globe=None)

    col_map = 'coolwarm'
    if data_variable_name == 'extreme_yn':
        col_map = 'cet_CET_L19'
    

    visual = ds_ext[data_variable_name].isel(time=slice(start_day-1, end_day, step_n_days)).hvplot.quadmesh(
    'lon', 'lat', projection=proj, project=True, global_extent=False, geo=True, 
    cmap=col_map, rasterize=True, dynamic=False, coastline=True,
    frame_width=500)
    
    pane = pn.panel(visual)
    return pane


#### ####################################################################    
#### 3.8 load_1_yr_results_and_basic_analysis()
#### A function to analyze Heatwaves, extreme cold Temperature 
#### from 1-year data results
#### Returns - xarray dataset
#### ####################################################################

# specify filename to test:
def load_1_yr_results_and_basic_analysis(ext_filename, show_basic_analysis=True, area_of_interest=None):
    ds_ext = xr.open_dataset(ext_filename)

    if not show_basic_analysis:
        return 
    
    # specify range of days to test:
    print_attrs = ['region_id', 'Analysis year', 'threshold', 'based_on_averages'
                   , 'is_percentage', 'averages_start_year', 'averages_end_year']
    attr_info = 'Extreme Temperature Base Results for: \n'
    for attribute in print_attrs:
        value = str(ds_ext.attrs[attribute])        
        attr_info += '{}: {}; '.format(attribute, value)
    print(attr_info)
    print('** from the filename: ', ds_ext.attrs['Store as'], '**')

    if area_of_interest is not None:
        ds_ext = region_subset(ds_ext, area_of_interest.get('top_lat'), area_of_interest.get('bottom_lat')
                               , area_of_interest.get('left_lon'), area_of_interest.get('right_lon'))
        
    return ds_ext

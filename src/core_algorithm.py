
# from matplotlib import pyplot as plt
# import matplotlib.dates as mdates
# from matplotlib.patches import Rectangle

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import xarray as xr

import io
import os

# data management
import xarray as xr
import numpy as np
import pandas as pd

# utilities
import time
import urllib.parse


################################
# Utils
################################

def get_xy_meshgrid(arr_tmax3d:np.ndarray) -> np.ndarray:
    """create grid for all x,y coordinate pairs [0,1],[0,2],..[283,583]"""
    
    shape_yx = arr_tmax3d.shape[1:] # np.shape order -> zyx
    arr_y = np.arange(shape_yx[0])
    arr_x = np.arange(shape_yx[1])
    ac = np.array(np.meshgrid(arr_x, arr_y)).T.reshape(-1, 2)
    
    return ac

def print_stats(arr:np.ndarray) -> None:
    size = round(arr_tmax3d.nbytes/1e9,2)
    shp = arr_tmax3d.shape
    print(f"""processing.. year={year}, shape z,y,x={shp}, in-memory={size} GB""")


################################
# NEW algorithm that uses Ravi's Avg Temps
################################

# input: 1D tmax data
# output: same-size 1D flags where heat event == True
    
def flag_heat_events(arr_tmax1d: np.array, arr_tavg1d:np.array, temp_thresh:int, days_thresh:int) -> np.array:
    """Feed 1 calendar year of data at a time."""
    
    # enrich
    df = pd.DataFrame({'diff': arr_tmax1d - arr_tavg1d})
    df['hot'] = df['diff'] > temp_thresh 
    df['label'] = df['hot'].diff().ne(False).cumsum()
    df = df.reset_index().reset_index()

    # filter
    df['isSummer'] = (121 < df.index) & (df.index < 273) # May1-Sep1 
    dff = df[df['isSummer'] & df['hot']].dropna(subset=['diff']) 
    
    # groupby
    dfg = dff.groupby('label').agg({
        'index':[np.min,np.max,len],
        'diff':np.max
    })
    dfg.columns = ['i1','i2','count','peak_diff']
    dfg = dfg[dfg['count'] >= days_thresh]
    dfg = dfg.drop('count', axis=1)
    dfg = dfg.reset_index(drop=True)

    # explode flags to a 365-length array
    arr = np.empty(len(df), dtype=np.float64)
    arr[:] = np.nan
    for _, (i, j, peak_diff) in dfg.iterrows():
        arr[int(i):int(j)+1] = peak_diff
        
    return arr

############### INPUTS ######################
lat_min = 0   
lat_max = 50  
lon_min = 220 
lon_max = 300 

TEMP_DIFF_THRESHOLD = 2 # Celcius (or K)
PERSISTED_FOR = 3 # days

years = range(1970,1980)
############################################# 

area = dict(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

arr_tavg3d = ds_avg['tasmaxavg'].sel(**area).values # 365 fixed lenght


for year in years:

    # temp data
    arr_tmax3d = ds['tasmax'].sel(time=str(year)).sel(**area).values

    # empty array to populate
    arr_heat3d = np.empty(arr_tmax3d.shape, dtype=np.float64)
    arr_heat3d[:] = np.nan

    # loop
    meshgrid = get_xy_meshgrid(arr_tmax3d)
    for i, j in meshgrid:
        
        arr_tmax1d = arr_tmax3d[:,j,i]

        if np.isnan(arr_tmax1d).all():
            arr_heat1d = np.empty(arr_tmax1d.shape, dtype=np.float64)
            arr_heat1d[:] = np.nan
        else:
            arr_tavg1d = arr_tavg3d[:,j,i]
            arr_heat1d = flag_heat_events(arr_tmax1d, arr_tavg1d, TEMP_DIFF_THRESHOLD, PERSISTED_FOR)

        arr_heat3d[:,j,i] = arr_heat1d  

    np.save(f'Koray/CMIP5_flagged/arr_heat3d-{year}.npy', arr_heat3d)
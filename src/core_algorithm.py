
import numpy as np
import pandas as pd


def get_xy_meshgrid(arr_tmax3d:np.ndarray) -> np.ndarray:
    """create grid for all x,y coordinate pairs [0,1],[0,2],..[283,583]"""
    
    shape_yx = arr_tmax3d.shape[1:] # np.shape order -> zyx
    arr_y = np.arange(shape_yx[0])
    arr_x = np.arange(shape_yx[1])
    ac = np.array(np.meshgrid(arr_x, arr_y)).T.reshape(-1, 2)
    
    return ac

def flag_heat_events(arr_tmax1d: np.array, arr_tavg1d:np.array, temp_thresh:int, days_thresh:int) -> np.array:
    """Feed 1 calendar year of data at a time.
    # input: 1D tmax data
    # output: same-size 1D flags where heat event == True
    """
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


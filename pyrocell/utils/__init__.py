import pandas as pd
import numpy as np

def load_data(path: str) -> tuple:
    """
    Loads experiment data from a csv file. This file must have:
    - Time (h) column
    - Cell columns, name starting with 'Cell'
    - Background columns, name starting with 'Background'

    :param str path: Path to the csv file.

    :return tuple: Split, formatted experimental data
    - time: time in hours
    - bckgd: background time-series data
    - backgd_length: length of each background trace
    - M: count of background regions
    - y_all: cell time-series data
    - y_length: length of each cell trace
    - N: count of cell regions
    """
    df = pd.read_csv(path).fillna(0)  
    data_cols = [col for col in df if col.startswith('Cell')]
    bckgd_cols = [col for col in df if col.startswith('Background')]
    time = df['Time (h)'].values[:,None]

    bckgd = df[bckgd_cols].values
    M = np.shape(bckgd)[1]
    
    bckgd_length = np.zeros(M,dtype=np.int32)
    
    for i in range(M):
        bckgd_curr = bckgd[:,i]
        bckgd_length[i] = np.max(np.nonzero(bckgd_curr))
        
    y_all = df[data_cols].values  
    
    N = np.shape(y_all)[1]
    
    y_all = df[data_cols].values
    np.max(np.nonzero(y_all))
    
    y_length = np.zeros(N,dtype=np.int32)
    
    for i in range(N):
        y_curr = y_all[:,i]
        y_length[i] = np.max(np.nonzero(y_curr))    

    return time, bckgd, bckgd_length, M, y_all, y_length, N
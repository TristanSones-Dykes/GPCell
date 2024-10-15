# External library imports
import pandas as pd
import torch

# External type imports
from torch import Tensor

def load_data(path: str) -> tuple[Tensor, Tensor, Tensor, int, Tensor, Tensor, int]:
    """
    Loads experiment data from a csv file. This file must have:
    - Time (h) column
    - Cell columns, name starting with 'Cell'
    - Background columns, name starting with 'Background'

    :param str path: Path to the csv file.

    :return Tuple[Tensor, Tensor, Tensor, int, Tensor, Tensor, int]: Split, formatted experimental data
    - time: time in hours
    - bckgd: background time-series data
    - bckgd_length: length of each background trace
    - M: count of background regions
    - y_all: cell time-series data
    - y_length: length of each cell trace
    - N: count of cell regions
    """
    df = pd.read_csv(path).fillna(0)  
    data_cols = [col for col in df if col.startswith('Cell')]
    bckgd_cols = [col for col in df if col.startswith('Background')]
    time = torch.from_numpy(df['Time (h)'].values[:,None])

    bckgd = torch.from_numpy(df[bckgd_cols].values)
    M = bckgd.shape[1]
    
    bckgd_length = torch.zeros(M,dtype=torch.int32)
    
    for i in range(M):
        bckgd_curr = bckgd[:,i]
        bckgd_length[i] = torch.max(torch.nonzero(bckgd_curr))
        
    y_all = torch.from_numpy(df[data_cols].values)
    
    N = y_all.shape[1]
    
    y_length = torch.zeros(N,dtype=torch.int32)
    
    for i in range(N):
        y_curr = y_all[:,i]
        y_length[i] = torch.max(torch.nonzero(y_curr))

    return time, bckgd, bckgd_length, M, y_all, y_length, N
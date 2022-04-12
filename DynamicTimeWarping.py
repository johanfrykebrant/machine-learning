import numpy as np
from numpy import shape
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html

def print_matrix(m):
    for row in m:
        print(row)

def DTW(s,t,window = np.inf):
    """Compares the similarity two vectors containing some sort of time series data"""
    n , m = len(s), len(t)
    # Adapting the window constraint 
    # If the window is shorter than the difference in lenght of the vectors, the start and end of the vectors will not be able to match up.
    w = max(window, abs(n-m))
    dtw_matrix = np.ndarray(shape = (n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i,j] = np.inf
    
    dtw_matrix[0,0] = 0

    for i in range(1,n+1):
        for j in range(max(1,i-w),min(m,i+w)+1):
            dtw_matrix[i,j] = 0  
    
    for i in range(1,n+1):
        for j in range(max(1,i-w),min(m,i+w)+1):
            distance = abs(s[i-1] - t[j-1])
            left = dtw_matrix[i-1,j]
            over = dtw_matrix[i,j-1]
            diagonal = dtw_matrix[i-1,j-1]
            prev_min = min(left,over,diagonal)
            dtw_matrix[i,j] = distance + prev_min
    
    return dtw_matrix[i][j]

if __name__ == "__main__":
    x1 = np.arange(1, 11, 0.1)
    x2 = np.arange(0, 10, 0.1)
    y1 = np.sin(x1)
    y2 = np.sin(x2)

    print(DTW(y1,y2))

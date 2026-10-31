import numpy as np
import ot
from scipy.spatial.distance import cdist

def cal_2_wasserstein_dist(X: np.ndarray, Y: np.ndarray) -> float:
    # X, Y shape: (N, 2)
    # 2-Wasserstein Distance = sqrt( Earth Mover's Distance with squared Euclidean cost )
    
    # 1. 建立 Cost Matrix (歐式距離的平方)
    M = cdist(X, Y, metric='sqeuclidean')
    
    # 2. 設定分佈權重 (假設是均勻分佈)
    a = np.ones((X.shape[0],)) / X.shape[0]
    b = np.ones((Y.shape[0],)) / Y.shape[0]
    
    # 3. 計算 EMD (這會回傳 cost 的加權總和)
    wd_sq = ot.emd2(a, b, M)
    
    # 4. 開根號
    return np.sqrt(wd_sq)


def cal_energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    # Energy Distance: 2 * E||X-Y|| - E||X-X'|| - E||Y-Y'||
    # 這裡用的是原始歐式距離 (Euclidean)
    
    dist_xy = cdist(X, Y, metric='euclidean').mean()
    dist_xx = cdist(X, X, metric='euclidean').mean()
    dist_yy = cdist(Y, Y, metric='euclidean').mean()
    
    return 2 * dist_xy - dist_xx - dist_yy
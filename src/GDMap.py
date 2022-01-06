import numpy as np

def kernal_mat(Us):
    n = Us.shape[0]
    
    ks = np.empty((n,n))
    
    for i in range(n):
        for j in range(n):
            ks[i,j] = np.linalg.norm(Us[i].T @ Us[j], ord = 'fro')**2
            
    return ks

def GDMap(U_array, rank):
    r = rank
    N = U_array.shape[0]
    m = U_array.shape[1]
    n = U_array.shape[2]
    
    Us = np.empty((N, m, r))
    Vs = np.empty((N, n, r))
    for i in range(N):
        u,s,vt = np.linalg.svd(U_array[i], full_matrices = False)
        
        Us[i] = u
        Vs[i] = vt.T
        
    ks_U = kernal_mat(Us)
    ks_V = kernal_mat(Vs)
    
    ks = ks_U + ks_V
    
    D_ii = np.empty(N)
    D_jj = np.empty(N)
    for i in range(N):
        D_ii[i] = sum(ks[i,:])
        D_jj[i] = sum(ks[:,i])
        
    ks_norm = np.empty(ks.shape)
    for i in range(N):
        for j in range(N):
            ks_norm[i,j] = ks[i,j] / np.sqrt(D_ii[i]* D_jj[j])
            
    P = np.empty(ks_norm.shape)
    for i in range(N):
        for j in range(N):
            P[i,j] = ks_norm[i,j] / sum(ks_norm[i,:])
            
    u,s,vt = np.linalg.svd(P, full_matrices = False)
    
    embedding = np.empty((N,2))
    for i in range(N):
        embedding[i,0] = s[0] * u[i,0]
        embedding[i,1] = s[1] * u[i,1]
        
    return embedding
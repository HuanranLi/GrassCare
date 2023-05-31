import numpy as np


'''
Geodesic Distance for Grassmannian
'''
'''
def d_G(U_i, U_j):
   
    
    u,s,v = np.linalg.svd(U_j @ U_j.T @ U_i)
    #A = U_j @ U_j.T @ U_i
    #u,v = np.linalg.eig(A.T@A)
    #s = np.sqrt(abs(u))
  
   
 

    for i in range(len(s)):
        if s[i] - 1 > 1e-5:
            raise Exception('s[',i,'] = ', s[i])
        elif s[i] > 1:
            s[i] = 1

    distance = np.sqrt( np.sum( np.arccos(s)**2 ))

    return distance
'''
def d_G(U_i, U_j): 
    s = np.linalg.svd(U_j @ U_j.T @ U_i, full_matrices=False, compute_uv=False)
    s = np.clip(s, 0, 1)
    distance = np.linalg.norm(np.arccos(s))
    return distance


'''
Geodesic Distance Gradient for Grassmannian
'''
'''
def del_d_G(U_i, U_j):
    m = U_i.shape[0]
    r = U_i.shape[1]
    assert U_j.shape == U_i.shape

    v, s, wt = np.linalg.svd(U_j @ U_j.T @ U_i)
    dg_UU = np.zeros((m,r))

    for l in range(r):
        if s[l] < 1:
            dg_UU += -1 * np.arccos(s[l]) / np.sqrt(1 - s[l]**2) * np.outer(v[:, l] , wt[l, :])
        else:
            dg_UU += -1 * np.outer(v[:, l] , wt[l, :])

    #cap singular values
    for l in range(r):
        if s[l] - 1 > 1e-5:
            raise Exception('s[',l,'] = ', s[l])
        elif s[l] > 1:
            s[l] = 1

    distance = np.sqrt( np.sum( np.arccos(s)**2 ))

    if distance > 1e-10:
        dg_UU = (np.identity(m) - U_i @ U_i.T) / distance @ dg_UU
        return distance, dg_UU
    else:
        return distance, np.zeros((m,r))

'''
def del_d_G(U_i, U_j):
    m = U_i.shape[0]
    r = U_i.shape[1]

    v, s, wt = np.linalg.svd(U_j @ U_j.T @ U_i, full_matrices=False)
    dg_UU = np.zeros((m,r))
   
    mask = s < 1
    dg_UU -= np.arccos(s[mask]) / np.sqrt(1 - s[mask]**2) * np.outer(v[:, mask].sum(axis=1), wt[mask, :])

          
    # Cap singular values
    s = np.clip(s, 0, 1)

    #distance = np.sqrt( np.sum( np.arccos(s)**2 ))
    distance = np.linalg.norm(np.arccos(s))

    if distance > 1e-10:
        dg_UU = (np.identity(m) - U_i @ U_i.T) / distance @ dg_UU
        return distance, dg_UU
    else:
        return distance, np.zeros((m,r))



'''
Distance in 2-d Embedding. The calculation is based on parameter method
'''
'''
def d_p(p_i, p_j):
    dist = 1 + 2 * (p_i - p_j).T @ (p_i - p_j) / (1 - p_i.T @ p_i) / (1 - p_j.T @ p_j)
    assert dist >= 1
    return np.arccosh(dist)
'''
def d_p(p_i, p_j):
    diff = p_i - p_j
    numerator = 2 * np.dot(diff, diff.T)
    denominator = (1 - np.dot(p_i.T, p_i)) * (1 - np.dot(p_j.T, p_j))
    dist = 1 + numerator / denominator
    assert dist >= 1
    return np.arccosh(dist)



'''
Gradient of Distance in 2-d Embedding. The calculation is based on parameter method
'''
'''
def del_d_p(p_i, p_j):
    if (p_i - p_j).T @ (p_i - p_j) < 1e-8:
        return np.zeros(np.shape(p_i))

    alpha = 1 - p_i.T @ p_i
    beta = 1 - p_j.T @ p_j

    assert alpha != 0
    assert beta != 0

    gamma = 1 + 2 / alpha / beta * (p_i - p_j).T @ (p_i - p_j)

    assert gamma > 1

    deri = 4 / beta / np.sqrt(gamma**2 - 1)
    deri *= (p_j.T @ p_j - 2 * p_i.T @ p_j + 1) / (alpha**2) * p_i - p_j / alpha

    return deri
'''



def del_d_p(p_i, p_j):
    diff = p_i - p_j
    squared_norm = np.dot(diff, diff.T)

    if squared_norm < 1e-8:
        return np.zeros(np.shape(p_i))

    alpha = 1 - np.dot(p_i.T, p_i)
    beta = 1 - np.dot(p_j.T, p_j)

    gamma = 1 + 2 / alpha / beta * squared_norm


    deri = 4 / beta / np.sqrt(gamma**2 - 1)
    deri *= (np.dot(p_j.T, p_j) - 2 * np.dot(p_i.T, p_j) + 1) / (alpha**2) * p_i - p_j / alpha

    return deri


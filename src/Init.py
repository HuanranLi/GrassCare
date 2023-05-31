import numpy as np
'''
Initialize a group of orthonormal vectors/matrix. If the clusters > 0, then the
points will be distributed in several clusters. Otherwise, points will be
distributed randomly.

Parameter:
1. vector/matrix dimension: ambient_dimension * rank
2. count: the number of points the caller want to initialized
3. clusters: the number of clusters the caller want to initialized
4. bound_zero: if x < bound_zero, it will be treated as 0
5. err_var: the variance of the clusters
'''
def U_array_init(ambient_dimension, rank, count, clusters, bound_zero = 1e-10,  err_var = 0.1):
    m = ambient_dimension
    r = rank

    if clusters == 0:
        U_array = [np.random.randn(m,r) for i in range(count)]
        for i in range(count):
            if r == 1:
                U_array[i] /= np.linalg.norm(U_array[i])
                #if m == 3:
                    #U_array[i][2] = abs(U_array[i][2])
            else:
                q_i,r_i = np.linalg.qr(U_array[i])
                U_array[i] = q_i

        return np.array(U_array),  abs(np.array(U_array)[:,2,0]) , None
        
    else:
        labels = []
        assert count % clusters == 0

        #calculate points per cluster
        n = count // clusters

        U_array = [np.random.randn(m,r) for i in range(clusters)]
        new_U_array = []
        for i in range(clusters):
            if r == 1:
                U_array[i] /= np.linalg.norm(U_array[i])
                if m == 3:
                    U_array[i][2] = abs(U_array[i][2])
            else:
                q_i,r_i = np.linalg.qr(U_array[i])
                U_array[i] = q_i

            #make sure its orthogonal
            #print(np.linalg.norm(U_array[i].T @ U_array[i] - np.identity(r)))
            assert np.linalg.norm(U_array[i].T @ U_array[i] - np.identity(r)) < bound_zero
            #make sure its normal
            assert  np.linalg.norm( np.linalg.norm(U_array[i], axis = 0) - np.ones(r) )  < bound_zero


            #generate points per cluster with parameter err_var
            for j in range(n):
                new_U = U_array[i] + np.random.randn(U_array[i].shape[0], U_array[i].shape[1])* err_var
                u,s,vt = np.linalg.svd(new_U, full_matrices= False)

                new_U_array.append(u@vt)

                if r == 1 and m == 3:
                    new_U_array[-1][2] = abs(new_U_array[-1][2])

                labels.append(i)


    return np.array(new_U_array), np.array(labels), np.array(U_array)
    
    
def U_array_3d_init(count):
    # Number of points to generate
    num_points = int(np.round(np.sqrt(count)))

    # Generate angles
    theta = np.linspace(0.01, 2*np.pi, num_points)
    phi = np.linspace(0.01, np.pi, num_points)

    # Create a meshgrid from angles
    theta_mesh, phi_mesh = np.meshgrid(theta, phi)

    # Convert spherical coordinates to Cartesian coordinates
    x = np.sin(phi_mesh) * np.cos(theta_mesh)
    y = np.sin(phi_mesh) * np.sin(theta_mesh)
    z = np.cos(phi_mesh)
    
    U_array = np.column_stack((x.flatten(), y.flatten(), z.flatten())).reshape(num_points**2, 3, 1)

    
    return U_array,  abs(np.array(U_array)[:,2,0])
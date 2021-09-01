import numpy as np
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1,'./src')

from grasscare import *




def main():
    ambient_dimension = m = 20
    rank = r = 2
    observed_vectors = n_K = 50

    missing_percentages = [0,0.5,]

    S, labels, optional_params = GROUSE_demo_init(m, r, n_K, missing_percentages)

    #optional_params['video_tail'] = 20
    embedding, info = grasscare_plot(S = S, labels = labels, video = True, optional_params = optional_params)

    #print(embedding)
    #print(embedding.shape)



def GROUSE_demo_init(m, r, n_K, missing_percentages):
    U_array, labels, centers = U_array_init(ambient_dimension = m, rank = r, count = 1, clusters = 1)

    print('\n########## GROUSE initialization ##########')
    GROUSE_paths = []
    path_names = []

    labels = np.array([i for i in range(len(missing_percentages))])
    #initializing multiple GROUSE path with same starting point and target but different missing_percentage
    for i in range(len(missing_percentages)):
        path_names.append(str(int(missing_percentages[i] * 100))  + '% missing')

        if i == 0:
            GROUSE_dict = GROUSE_init(U_array, missing_percentage = missing_percentages[i], observed_vectors = n_K)
        else:
            GROUSE_dict = GROUSE_init(U_array, missing_percentage = missing_percentages[i], observed_vectors = n_K,
                                        U_0 = GROUSE_paths[0][0], U_0_load = True)

        GROUSE_paths.append(GROUSE(GROUSE_dict, max_iter = 1, eta = 1))

        print('missing Percentage:', missing_percentages[i])
        print('\tDistance(U*, U_t) = ', d_G(U_array[0], GROUSE_paths[-1][-1]))

    optional_params = { 'path_names': path_names, 'Targets': U_array}

    GROUSE_paths = np.array(GROUSE_paths)
    S = []
    for t in range(n_K):
        S.append(GROUSE_paths[:,t])
    S = np.array(S)

    print('S Shape:', S.shape)

    return S, labels, optional_params




if __name__ == '__main__':
    main()

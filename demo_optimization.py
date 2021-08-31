
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1,'./src')


from Initialization import *
from Gradient_Descent import *
from grasscare import *
from GROUSE import *
from Plot_Functions import *

import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import seaborn as sns
import IPython


def main():
    ambient_dimension = m = 3
    rank = r = 1
    count = N = 30
    clusters = K = 3

    assert clusters > 0
    assert count % clusters == 0

    S, labels, centers = U_array_init(ambient_dimension = m, rank = r, count = N, clusters = K)

    optional_params = {'video_tail': 5}
    #visualizing Us
    if m == 3 and r == 1:
        plot_U_array(S, labels = labels, title = 'Initialization')

    embedding, info = grasscare_plot(S = S, labels = labels, video = True, optional_params = optional_params)




if __name__ == '__main__':
    main()

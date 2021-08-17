

<!-- PROJECT LOGO -->
<br />
<p align="center">


  <h3 align="center">Poincar&eacute; Embedded Symmetric SNE (PESSNE)</h3>

</p>




<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
      <ul>
           <li><a href="#function-interface">Function Interface</a></li>
      </ul>
      <ul>
           <li><a href="#demo">Demo</a></li>
      </ul>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
<BODY>
  <IMG SRC="https://github.com/HuanranLi/Poincare-Embedded-Symmetric-SNE/blob/main/graphs/Poincare_Poincare_s-SNE.gif" width="512" height="512">
</BODY>
  
This project provides a method similar to <a href = 'https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding'>t-SNE</a> for finding low-dimensional embedding for <a href = 'https://en.wikipedia.org/wiki/Grassmannian'>Grassmannian</a>. Instead of plotting it on to a 2-d plane, this algrithm manages to find the embedding on a 2-d <a href = 'https://en.wikipedia.org/wiki/Poincar%C3%A9_disk_model'>Poincar&eacute; Disk</a>, which is superior in representing points with hierarchical structures.

### Built With

The major framework this project uses are Python, and its supplementary packages.
* [Python3](https://www.python.org/)
* [Numpy](https://numpy.org/)
* [Matplotlib](https://matplotlib.org/)

## Usage
### Function Interface
<!-- Function Interface -->
 The main function is under [src/grasscare.py](https://github.com/HuanranLi/GrassCare-Plot/blob/main/src/grasscare.py), called `grasscare_plot(S, labels, video, optional_params = {})`. An embedding will be calculated and returned in the same format as the input array S. A graph will be plotted containing all the points with the new embedding. It has following parameters:
  * S: A matrix of Grassmannian points, where U_i^t means i'th Grassmannian point at time t.
  
    <table id="vertical-1">
            <tr>
                <th>t = 0</th>
                <td>U<sub>0</sub><sup>0</sup></td> <td>U<sub>1</sub><sup>0</sup></td> <td>...</td> <td>U<sub>1</sub><sup>0</sup></td> 
            </tr>
            <tr>
                <th>t = 1</th>
                <td>U<sub>0</sub><sup>1</sup></td> <td>U<sub>1</sub><sup>1</sup></td> <td>...</td> <td>U<sub>1</sub><sup>1</sup></td> 
            </tr>
            <tr>
                <th>...</th>
                      <td>...</td>      <td>...</td>      <td>...</td>      <td>...</td>
            </tr>
            <tr>
                <th>t = T</th>
                <td>U<sub>0</sub><sup>T</sup></td> <td>U<sub>1</sub><sup>T</sup></td> <td>...</td> <td>U<sub>1</sub><sup>T</sup></td> 
            </tr>
        </table>
  
  * labels: A np.ndarray with N elements where i'th entry corresponds to U_i'th label. Labels start index at 0.
  * video: True - Generate a video in gif format to display the path with shifting in time. If only one single time frame is contained in S, (i.e. T = 0), video will be generated to show its optimization path instead of clustering path.
  * optional_params: It is default to be an empty dictionary. If no optional parameter is needed, just run `grasscare_plot(S, labels, video)`. Here are optional parameters:
    <table id="OPT">
    <tr>
      <th>Name</th>
      <th>Value Type</th>
      <th>Explanation</th>
    </tr>
    <tr>
      <td>video_tail</td>
      <td>Int</td>
      <td>The length of the tail drawn in the video. (input -1 if all tail points needed to be drawn). Default: -1</td>
    </tr>
    <tr>
      <td>b_array_init_syle</td>
      <td>{'PCA', 'random'}</td>
      <td>Define how the initial low-dimensional points are created. 'PCA' would vectorize the Grassmannian matrix and do PCA. Default: 'random'</td>
    </tr>
     <tr>
      <td>max_epoch</td>
      <td>int</td>
      <td>The maximum of iteration for the optimization process. Default: 500</td>
    </tr>
     <tr>
      <td>step_size</td>
      <td>Float</td>
      <td>The step size for the optimization process. Default: 1</td>
    </tr>
     <tr>
      <td>objective_plot</td>
      <td>Boolean</td>
      <td>Plot the objective function at the end of optimization. Default: True</td>
    </tr>     
    <tr>
      <td>Targets</td>
      <td>np.ndarray([Matrices])</td>
      <td>An array of targets points wanted to be drawn if available. Default: []</td>
    </tr>
    <tr>
      <td>path_names</td>
      <td>np.ndarray([Strings])</td>
      <td>An array of names for each point, if a legend is wanted to be drawn. Mainly used for GROUSE. Default: []</td>
    </tr>
    </table>
  
  



<!-- USAGE EXAMPLES -->
### Demo
* To find the best embedding for clusters' path, please refer [demo_clusterpath.py](https://github.com/HuanranLi/GrassCare-Plot/blob/main/demo_clusterpath.py). It can be run by: `python3 demo_clusterpath.py`
* To find the best embedding for GROUSE routes, please refer [demo_GROUSE.py](https://github.com/HuanranLi/GrassCare-Plot/blob/main/demo_GROUSE.py). It can be run by: `python3 demo_GROUSE.py`
* To see how the optimization process is done, please refer [demo_optimization](https://github.com/HuanranLi/GrassCare-Plot/blob/main/demo_optimization.py). It can be run by: `python3 demo_optimzation.py`
  
<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/HuanranLi/Poincare-Embedded-Symmetric-SNE/issues) for a list of proposed features (and known issues).




<!-- CONTACT -->
## Contact

Huanran Li - [Website](https://huanranli.github.io/) - hli488@wisc.edu

Project Link: [https://github.com/HuanranLi/GrassCare-Plot](https://github.com/HuanranLi/GrassCare-Plot)





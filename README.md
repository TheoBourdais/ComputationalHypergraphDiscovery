<img style="width:100%;" alt="CHD logo" src="images/logo.jpg"></a>


# Computational Hypergraph Discovery: A Gaussian process framework for connecting the dots

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11.4](https://img.shields.io/badge/python-3.11.4-blue.svg)](https://www.python.org/downloads/release/python-3114/)
[![last commit](https://img.shields.io/github/last-commit/TheoBourdais/ComputationalHypergraphDiscovery.svg)](https://github.com/TheoBourdais/ComputationalHypergraphDiscovery/commits/main)
[![Cite this repository](https://img.shields.io/badge/Cite%20this%20repository-CITATION.cff-blue.svg)](https://github.com/TheoBourdais/ComputationalHypergraphDiscovery/blob/main/CITATION.cff)


release todo:
- make repo public
- add link to paper
- add link to blog post
- create citation.cff
- modify blog post with paper link
- add sachs data to repo, and modify code in readme and blog post to use it,
- make sachs data a csv with column title
- modify gene code to fit new interface
- modify plot code to see individual activations

This is the source code for the paper ["Computational Hypergraph Discovery: A Gaussian process framework for connecting the dots"][paper_url]. 

Please see the [companion blog post][blog_url] for a gentle introduction to the method and the code.


## Installation 

The code is written in Python 3 and requires the following packages:
- matplotlib
- numpy
- scipy
- scikit-learn
- networkx

After cloning this repo, you install these packages using pip:
```bash
pip install -r requirements.txt
```


## Quick start

Graph discovery takes very little time. The following code runs the method on the example dataset provided in the repo. The dataset is a 2D array of shape (n_samples, n_features) where each row is a sample and each column is a feature. After fitting the model, the graph is stored in the `GraphDiscovery` object, specifically it's graph `G` attribute. The graph is a `networkx` object, which can be easily plotted using `.plot_graph()`.

```python
import ComputeHypergraphDiscovery as CHD
import pandas as pd
df=pd.read_csv('examples\sachs.csv')
kernel=CHD.Modes.LinearMode()+CHD.Modes.QuadraticMode()
graph_discovery = CHD.GraphDiscovery.from_dataframe(df,mode_kernel=kernel)
graph_discovery.fit()
graph_discovery.plot_graph()
```

You should obtain the following graph:

<img style="width:100%;" alt="Resulting graph Sachs example" src="examples\sachs.png"></a>


## Available modifications of the base algorithm

The code gives an easy-to-use interface to manipulate the graph discovery method. It is designed to be modular and flexible. The main changes you can make are
- **[Kernels and modes](#manipulating-kernels)**: You can decide what type of function will be used to link the nodes. The code provides a set of kernels, but you can easily add your own. The interface is designed to ressemble the scikit-learn API, and you can use any kernel from scikit-learn. 
- **[Decision logics](#manipulating-decision-logics)**: In order to identify the edges of the graph, we need to decide wether certain connections are significant. The code provides indicators (like level of noise) and the user specifies how to interpret them. The code provides a set of decision logics, but you can define your own. 
- **[Clustering](#manipulating-clusters-and-possible-edges)**: If a set of nodes is highly dependent, it is possible to merge them into a cluster of nodes. This gives greater readability and prevents the graph discovery method from missing other connections. 
- **[Possible edges](#manipulating-clusters-and-possible-edges)**: If you know that certain nodes cannot be connected, you can specify it to the algorithm. By default all edges are possible. 


## The Base algorithm

In order to initiate the graph discovery method, you need to create a `GraphDiscovery` object. The main parameters are:
- **Essential parameters**:
    - `X`: The data used to fit the model. It is a 2D array of shape (n_features,n_samples) where each row is a feature and each column is a sample.
    - `node_names`: The names of the nodes. It is a list of strings of length n_features.
- **Refinements**:
    - `mode_kernel`/`mode_container`: The kernel used to link the nodes. If none is provided, a default kernel is used. See the section on [manipulating kernels](#manipulating-kernels) for more details.
    - `cluster` and `possible_edges`: additional features to inform the graph discovery process and refine the recovery. See the section on [manipulating clusters and possible edges](#manipulating-clusters-and-possible-edges) for more details.
- **Optional**:
    - `verbose`: to choose wether to print information during the fitting process.


Here is an example:


```python
import ComputeHypergraphDiscovery as CHD
import numpy as np
X=np.random.rand(10,100)
node_names=[f'node_{i}' for i in range(10)]
graph_discovery = CHD.GraphDiscovery(X,node_names)
```

> **Note**: This shows how to create a `GraphDiscovery` object from a numpy array. If you want to follow this tutorial, use the sachs dataset provided in the repo as a reference. You can use the code below to load it:

If you have a Pandas dataframe, you can use the `from_dataframe` method (see the method's docstring for more details):

```python
import ComputeHypergraphDiscovery as CHD
import pandas as pd
df=pd.read_csv('examples\sachs.csv')
graph_discovery = CHD.GraphDiscovery.from_dataframe(df)
```

Once the object is created, you can fit the model using the `.fit()` method. The fit method takes the following parameters: 
- **Essential parameters**:
    - `targets`: Which nodes you wish to discover the ancestors of. By default, we recover the ancestors of all nodes. 
- **Refinements**:
    - `kernel_chooser`, `mode_chooser` and `early_stopping`: Decision logics to refine the graph discovery process. See the section on [manipulating decision logics](#manipulating-decision-logics) for more details.
- **Optional** (better left as default):
    - `gamma`: The noise parameter of the kernel. By default, it is automatically estimated. We advise against setting it manually, as a good choice of gamma is crucial for the performance of the algorithm, and unintuitive to find.
    - `gamma_min`: The minimum value of gamma when finding it automatically. By default, it is `1e-9`. This is necessary for numerical stability.

Here is an example:
```python
graph_discovery.fit()
```
If you wish to discover the ancestors of a specific node, you can do so by specifying the `targets` parameter:
```python
graph_discovery.fit(targets=['$Raf$'])
```

Once the model is fitted, you can access all the results of the graph discovery using the `.G` attribute. It is a `networkx` object, and you can use any of the methods provided by `networkx`. For example, you can plot the graph using the `.plot_graph()` method:
```python
graph_discovery.plot_graph()
```

>The `.plot_graph()` method allows for some customization of the resulting plot. See the method's docstring for more details on available parameters. 


## Manipulating kernels

The kernel is the function used to define the type of functions that will link the nodes. The code provides a set of kernels, but you can easily add your own. The interface is designed to ressemble the scikit-learn API, and you can use any kernel from scikit-learn. 

> **Definitions**: a kernel is a function $k$ such that $k(x,y)=\langle \phi(x),\phi(y)\rangle$ for some feature map $\phi$. it allos to define a similarity measure between two points and enhance the expressivity of the model. We use kernel extensively in this project, and will define some terms here:
> - **Kernel**: a function $k$ used to perforn the downstream ML task (here, identify the ancestors). 
> - **Kernel Matrix**: Given a kernel $k$ and a dataset $X$ of shape (n_features,n_samples), the kernel matrix is a matrix of shape (n_samples,n_samples) defined by $K_{i,j}=k(X_i,X_j)$.
> - **Kernel Mode**: A kernel mode is a kernel itself that is used to form a sum kernel (if $k=k_1+k_2$, $k_1$ and $k_2$ are kernel modes of $k$)
> - **Kernel Mode Matrix**: Given kernel modes $k_1,..,k_p$ of $k$ ($k=k_1+..+k_p$) and a dataset $X$ of shape (n_features,n_samples), the kernel mode matrix is a matrix of shape (p,n_samples,n_samples) defined by $K_{l,i,j}=k_l(X_i,X_j)$ where $k_l$ is the ith kernel mode of $k$. We have $K_{i,j}=k(X_i,X_j)=\sum_{l=1}^p K_{l,i,j}$. 

To manupulate kernels, import the modes module:
```python
import ComputationalHypergraphDiscovery.Modes as Modes
```


The `ModeKernel` interface defines our kernels modes. An example of a kernel mode is the `LinearMode` class, which implements the linear kernel that can be used as a mode. `ModeKernel` has two main components:
- **metadata**: all the necessary information that allow downstream algorithms to use the kernel and separate it into modes.
- **call function**: the `__call__` function takes an array of shape (n_features,n_samples) (typically the `X` variable that is given to the `GraphDiscovery` object) and returns the kernel mode matrix.

Let us detail the metadata

- `name`: the name of the kernel. It is used to identify the kernel.
- `hyperparameters`: a dictionary of hyperparameters. 
- `is_interpolatory`: whether the kernel is interpolatory or not. This is used to know how to compute the noise parameter gamma in `GraphDiscovery`.
- `mode_type`: the type of modes of the kernel. It is a string, and can be either `'individual'`, `'pairwise'` or `'combinatorial'`.


### Interpolatory: 

A kernel $k(x,y)=\langle\phi(x),\phi(y)\rangle$ will be interpolatory if its associated feature map $\phi$ has more dimensions than the number of data points. For example, the Gaussian kernel is interpolatory (its feature map is infinite dimensional), while the linear kernel will probably not be (if the input is of dimension d, the number of data points is n, if n>d the kernel is not interpolatory).

### Mode type: 
For our method to work, we need to be able to separate the kernel into modes (i.e. smaller kernels that we sum to get the full one) so that we can identify the contribution of each node into explaining their children nodes. To that endd, we've defined three types of modes:
- **Individual**: Each variable has its own mode. The kernel mode is applied to variables individually. Thus, if we have a kernel function $\kappa(x,y)$, and $X$ is a matrix of shape (n_features,n_samples), then the kernel mode matrix will be a matrix of shape (n_features,n_samples,n_samples). The ith matrix of shape (n_samples,n_samples) is the kernel $\kappa$ applied to the ith feature of $X$.
- **Pairwise**: Each pair of variable is a mode. Thus, if we have a kernel function $\kappa(x,y)$, and $X$ is a matrix of shape (n_features,n_samples), then the kernel mode matrix will be a matrix of shape (n_features,n_features,n_samples,n_samples). The matrix of shape (n_samples,n_samples) at index i,j is the kernel $\kappa$ applied to pairs of features (i,j) of $X$.
- **Combinatorial**: Each subset of variable of variable has a mode. Thus, if we have a kernel function $k(x,y)$, and $X$ is a matrix of shape (n_features,n_samples), then the kernel mode matrix will be of shape ($2^{n_{features}}$,n_samples,n_samples). For example, if we have 3 features, the and choose a kernel mode $\kappa$ to apply in a `'combinatorial'` fashion, the full kernel will be $\kappa(x_1,y_1)+\kappa(x_2,y_2)+\kappa(x_3,y_3)+\kappa(x_1,y_1)*\kappa(x_2,y_2)+\kappa(x_1,y_1)*\kappa(x_3,y_3)+\kappa(x_2,y_2)*\kappa(x_3,y_3)+\kappa(x_1,y_1)*\kappa(x_2,y_2)*\kappa(x_3,y_3)$. This is intractable to store and compute as is, however you can notice that $k(x,y)=\prod_{i=1}^3(1+\kappa(x_i,y_i))$. Thus, we can compute the kernel matrix of $\kappa$ for each feature, and then compute this product efficiently. The mode will yield a matrix of shape (n_features,n_samples,n_samples). The ith matrix of shape (n_samples,n_samples) is the kernel $\kappa$ applied to the ith feature of $X$. 

> Another way to understand this is to look at the code of how the kernel matrices are computed. In the custom modeKernel, here a simplified version of `__call__` function:
> ```python
> def __call__(self,X):
>    k=self.hyperparameters["scipy_kernel"]
>    n_features,n_samples=X.shape
>    if self.mode_type in ["individual", "combinatorial"]:
>        res = []
>        for col in X:
>            res.append(k(col))
>        matrix = np.stack(res, axis=0)
>        assert matrix.shape == (n_features, n_samples,n_samples)
>        return matrix
>    if self.mode_type == "pairwise":
>        res = np.zeros((n_features, n_features, n_samples,n_samples))
>        for i, col1 in enumerate(X):
>            for j, col2 in enumerate(X[: i + 1]):
>                data = np.stack([col1, col2], axis=1)
>                res[i, j, :, :] = k(data)
>        return res
>```
> Despite the fact that individual and combinatorial matrices are computed the same, they are not treated equally by the algorithm. To get the kernel matrix from the kernel mode matrix, we perform an operation like: 
> ```python
>def get_kernel_matrix(kernel_mode_matrix, matrix_type):
>   if matrix_type == "individual":
>       return np.sum(kernel_mode_matrix, axis=0)
>   if matrix_type == "pairwise":
>       return np.sum(kernel_mode_matrix,axis=(0,1))
>   if matrix_type == "combinatorial":
>       return np.prod(1 + kernel_mode_matrix, axis=0)
> ```

We provide a set of kernel modes:
- **LinearMode**: The linear kernel. 
    - not interpolatory 
    - `'individual'` 
    - $\kappa(x,y)=x*y$ (only implemented for $x,y\in\mathbb{R}$).
    - Call: `Modes.LinearMode()`
- **QuadraticMode**: The quadratic kernel. 
    - not interpolatory 
    - `'pairwise'` 
    - $\kappa((x_1,x_2),(y_1,y_2))=x_1x_2y_1y_2$ (only implemented for $x_1,y_1,x_2,y_2\in\mathbb{R}$).
    - Call: `Modes.QuadraticMode()`
- **GaussianMode**: The Gaussian kernel. 
    - interpolatory 
    -  `'combinatorial'` 
    - has hyperparameter `l` the lengthscale
    - $\kappa(x,y)=\exp(-(x-y)^2/(2l^2))$ only implemented for $x,y\in\mathbb{R}$.
    - Call: `Modes.GaussianMode(l=1)`
- **SklearnMode**: A wrapper around any kernel from scikit-learn. 
    - interpolatory to be specified
    - Type to be specified
    - $\kappa$ to be specified, must behave like Sklearn kernel. 
    - Call: `Modes.SklearnMode(sklearn_kernel,mode_type,is_interpolatory,name)`
- **PrecomputedMode**: A wrapper around a precomputed mode kernel matrix. 
    - interpolatory to be specified
    - Type to be specified
    - $\kappa$ to be specified as a pre-computed kernel mode matrix.
    - Call: `Modes.PreComputedMode(matrix,mode_type,is_interpolatory,name)`

### Combining kernels
>`ModekernelList` is more of an internal class, so you may not need to use it. However, it is useful to know what happens when you combine kernels with the `+`, as it is not the same as summing functions.

In order to be able to combine kernels, we have the `ModeKernelList` class, that allows to manipulate a list of kernel modes. It essentially behaves like a list, and **addition corresponds to concatenation**.  We also implemented multiplication with a scalar, which allows to scale kernels. The main use of this class is to combine modes into a sum kernel, using
```python
kernel=0.1*Modes.LinearMode()+0.01*Modes.QuadraticMode()+0.001*Modes.GaussianMode(l=1)
```
> **Caution**: We have defined here a kernel with 3 modes, one linear, one quadratic and one Gaussian. However, for the overall algorithm to work, we prefer to define the 3 following hierarchical modes:
> - Linear mode
> - Linear + quadratic mode
> - Linear + quadratic + Gaussian mode
>
> This allows to have modes of increasing complexity, so that when we change mode, the new one is strictly more expressive than the previous one (this means we widen the search as we progress)

>**Note**: You may be thinking: *there are modes everywehre!*, this is the point ! We can split our final kernel into as many modes as we wish, but specifically:
> - We can split the kernel into modes associated with one kind of functions (linear, quadratic, Gaussian, etc.)
> - We can split the kernel into modes associated with one specific variable. Indeed, each of the modes (linear, quadratic, gaussian) can be split into variable modes itself. 

Once your kernel is chosen, you can use it by specifying the `mode_kernel` parameter of the `GraphDiscovery` object. For example, if you want to use the linear kernel and quadraatic kernels: 
```python
import ComputationalHypergraphDiscovery as CHD
import ComputationalHypergraphDiscovery.Modes as Modes
import numpy as np
X=np.random.rand(10,100)
node_names=[f'node_{i}' for i in range(10)]
kernel=Modes.LinearMode()+Modes.QuadraticMode()
graph_discovery = CHD.GraphDiscovery(
    X=X,
    node_names=node_names,
    mode_kernel=kernel
    )
```

### Computing the mode kernel matrices and storing them

>This section concerns the internals of the code, and is not necessary to understand to use the code. You may need to look at `ModeContainer` to reuse matrix computations from one Graphdsicovery object to another.

Once we have defined our kernel, we need to compute and the kernel mode matrices. This is done by the `ModeContainer` class. It takes a kernel and a dataset, and computes the kernel mode matrices. It also stores them, and performs the necessary operations to retrieve the kernel matrix from the kernel mode matrices, especially when we start removing ancestors etc. 

## Manipulating decision logics

In order to identify the edges of the graph, we need to decide wether certain connections are significant. The code provides indicators (like level of noise) and the user specifies how to interpret them. The functions used to make these decisions are available in the `decision` module.

```python
import ComputationalHypergraphDiscovery.decision as decision
```

There are two main indicators that we use to make decisions (See the [paper][paper_url] or the [blog post][blog_url] for more details):
- **Signal to noise ratio**: The signal to noise ratio is a measure of how much the signal is stronger than the noise. We also use its complementary value the noise to signal ratio. Here it indicates the amount of noise we measure in the data given a set of ancestors. If there is a lot of noise this means the current set of ancestors doesn't explain well the node (and thus there arre not true ancestors). 
- **Z_test**: The Z_test is a statistical test that allows to test if the signal-to-noise ratio we obtained is statistically significant.

There are three types of decisions to be made in the algorithm:
- **Kernel chooser**: Chooses if an node has ancestors and with which kernel.
- **Mode chooser**: Chooses the ancestors of a mode
- **Early stopping**: Chooses to stop the pruning process early to save computation time.

### Kernel chooser
The first decision to make is to decide if a node has ancestors or not, as well as the kernel with which the node has ancestors. To do so, we compute the signal to noise ratio and the `Z_test` for each kernel. We get the following performances:

An example of output is the following dictionary (some keys have been hidden for clarity):
```python
kernel_performances={
  'linear': {
    'noise-to-signal ratio': 0.45,
    'Z_test': [0.98, 1.0]
    },
  'quadratic': {#see note above, this is not the quadratic kernel but linear+quadratic
    'noise-to-signal ratio': 0.91, 
    'Z_test': [0.96, 1.0]
    }, 
  'gaussian': {#same comment as above
    'noise-to-signal ratio': 0.33, 
    'Z_test': [0.74, 1.0]
    }
}
```

Severall decisions can be made from this dictionary. Decisions are made by instances of the `KernelChooser` class. 
- **Threshold**: 
    - Implemented in `ThresholdKernelChooser`. 
    - Choose an amount of noise that is acceptable, and choose the simplest kernel that has a noise-to-signal ratio below this threshold. 
    - If the threshold is too low, no kernel will be chosen 
    - In the example, with a threshold of 0.5 (i.e. half of the data is noise, half is signal), the linear kernel is chosen.
-  **Minimum noise**:
    - Implemented in `MinNoiseKernelChooser`.
    - Choose the kernel that has the lowest noise such that the noise is not in the `Z_test` interval (which would mean the noise ratio is not statistically significant).
    - If no kernel has a noise ratio that is not statistically significant, no kernel is chosen.
    - In the example, the gaussian kernel is chosen. 
- **Manual choice**:
    - Implemented in `ManualKernelChooser`.
    - Choose a kernel based on user input. 
    - In the example, if we choose the quadratic kernel, the quadratic kernel is subsequently used. 
- **Custom choice**:
    - Implemented in `CustomKernelChooser`.
    - Choose a kernel based on a custom function. 

>**Note on implementing custom kernel chooser**:
> - The `CustomKernelChooser` class allows you to put your own logic. You must provide a function `choice`:
>```python
>choice(kernel_performances:dict)->str
>``` 
>And the kernel performance dictionary has the following structure (irrelevant keys were masked for clarity):
>```python
>kernel_performances={
>  'linear': {
>    'noise': float, #the noise-to-signal ratio
>    'Z_test': (float, float) #the Z_test interval
>     ...
>    },
>  'quadratic': {
>    ...
>   },
>  ...
>}
>```
> Finally, the choice function must return the name of the chosen kernel as a string, or `None` if no kernel is chosen.
> Thus, the `choice` function is called in this way:
>```python
>class CustomKernelChooser(KernelChooser):
>    def __init__(self, chooser_function):
>        self.choice_function = chooser_function
>
>    def __call__(self, kernel_choice_dict):
>        res = self.choice_function(kernel_choice_dict)
>        assert (
>            res is None or res in kernel_choice_dict.keys()
>        ), f"invalid choice of kernel from custom function: {res}"
>        return res
>```

### Mode chooser

Once we have chosen a kernel, we need to choose the ancestors of the node. To do so, we remove the ancestors one by one, and compute the signal to noise ratio and the `Z_test` after each removal. The evolution of the signal to noise ratio and the `Z_test` gives the following graph: 

<img style="width:100%;" alt="Prototypical signal to noise ratio evolution" src="_images/signal_to_noise_ratio_evolution.png"></a>

As we remove ancestors, it becomes harder to explain the node from the remaining ancestors. Seeing the evolution of the signal to noise ratio and the `Z_test`, we must choose how many ancestors we wish to keep. 

>**Note**: One good indicator of when we have removed an essential ancestor is when we observe a spike in the noise ratio. This is why increments in the noise ratio are plotted on the right, as it is a good indicator. 

Severall decisions can be made from this evolution. By default we use `MaxIncrementModeChooser`. Decisions are made by instances of the `ModeChooser` class:
- **Threshold**: 
    - Implemented in `ThresholdModeChooser`. 
    - Choose a threshold for the noise ratio, and keep the ancestors until the noise ratio goes above this threshold.
    - In the example, with a threshold of 0.7, we would keep 8 ancestors.
- **Maximum increment**: 
    - Implemented in `MaxIncrementModeChooser`. 
    - We look at the highest spike in the noise ratio, and keep the ancestors we had at the moment of this spike.
    - In the example, we see the spike occurs at 2 ancestors, so we keep these 2 ancestors. 
- **Manual choice**:





## Manipulating clusters and possible edges

<!-- links -->

[paper_url]: https://example.com
[blog_url]: https://example.com

[go_download_url]: https://golang.org/dl/
[go_run_url]: https://pkg.go.dev/cmd/go#hdr-Compile_and_run_Go_program
[go_install_url]: https://golang.org/cmd/go/#hdr-Compile_and_install_packages_and_dependencies
[go_report_url]: https://goreportcard.com/report/github.com/gowebly/gowebly
[go_dev_url]: https://pkg.go.dev/github.com/gowebly/gowebly
[go_version_img]: https://img.shields.io/badge/Go-1.21+-00ADD8?style=for-the-badge&logo=go
[go_code_coverage_url]: https://codecov.io/gh/gowebly/gowebly
[go_code_coverage_img]: https://img.shields.io/codecov/c/gh/gowebly/gowebly.svg?logo=codecov&style=for-the-badge
[go_report_img]: https://img.shields.io/badge/Go_report-A+-success?style=for-the-badge&logo=none

<!-- Repository links -->

[repo_url]: https://github.com/gowebly/gowebly
[repo_issues_url]: https://github.com/gowebly/gowebly/issues
[repo_pull_request_url]: https://github.com/gowebly/gowebly/pulls
[repo_releases_url]: https://github.com/gowebly/gowebly/releases
[repo_license_url]: https://github.com/gowebly/gowebly/blob/main/LICENSE
[repo_license_img]: https://img.shields.io/badge/license-Apache_2.0-red?style=for-the-badge&logo=none
[repo_cc_license_url]: https://creativecommons.org/licenses/by-sa/4.0/
[repo_readme_ru_url]: https://github.com/gowebly/gowebly/blob/main/README_RU.md
[repo_readme_cn_url]: https://github.com/gowebly/gowebly/blob/main/README_CN.md
[repo_readme_es_url]: https://github.com/gowebly/gowebly/blob/main/README_ES.md
[repo_stargazers_url]: https://github.com/gowebly/gowebly/stargazers
[repo_badge_stargazers_img]: https://user-images.githubusercontent.com/11155743/275514241-8ecdf4bd-c35e-4e28-a937-b0a63aa1dbaf.png
[repo_default_config_url]: https://github.com/koddr/gowebly/blob/main/internal/attachments/configs/default.yml

<!-- Docs links -->

[docs_url]: https://gowebly.org
[docs_how_it_works_url]: https://gowebly.org/complete-user-guide/how-does-it-work/index.html
[docs_installation_url]: https://gowebly.org/complete-user-guide/installation/index.html
[docs_configuring_url]: https://gowebly.org/complete-user-guide/configuration/index.html
[docs_create_new_project_url]: https://gowebly.org/complete-user-guide/create-new-project/index.html
[docs_run_project_url]: https://gowebly.org/complete-user-guide/run-your-project/index.html
[docs_build_project_url]: https://gowebly.org/complete-user-guide/build-your-project/index.html

<!-- Author links -->

[author_url]: https://github.com/koddr

<!-- Readme links -->

[gowebly_helpers_url]: https://github.com/gowebly/helpers
[gowebly_youtube_video_url]: https://www.youtube.com/watch?v=qazYscnLku4
[gowebly_devto_article_url]: https://dev.to/koddr/a-next-generation-cli-tool-for-building-amazing-web-apps-in-go-using-htmx-hyperscript-336d
[cgapp_url]: https://github.com/create-go-app/cli
[cgapp_stars_url]: https://github.com/create-go-app/cli/stargazers
[htmx_url]: https://htmx.org
[hyperscript_url]: https://hyperscript.org
[brew_url]: https://brew.sh
[docker_image_url]: https://hub.docker.com/repository/docker/gowebly/gowebly
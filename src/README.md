
# Computational Hypergraph Discovery: A Gaussian process framework for connecting the dots

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/license/apache-2-0/)
[![Python 3.11.4](https://img.shields.io/badge/python-3.11.4-blue.svg)](https://www.python.org/downloads/release/python-3114/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-red)](https://github.com/TheoBourdais/ComputationalHypergraphDiscovery)

This is the source code for the paper ["Computational Hypergraph Discovery: A Gaussian process framework for connecting the dots"](https://arxiv.org/abs/2311.17007). 

Please see the [companion blog post](https://theobourdais.github.io/posts/2023/11/CHD/) for a gentle introduction to the method and the code. See the repo [here](https://github.com/TheoBourdais/ComputationalHypergraphDiscovery) for full documentation and examples.


## Installation 

The code is written in Python 3 and requires the following packages:
- matplotlib
- NumPy
- scipy
- scikit-learn
- networkx

You can install using pip:

```bash
pip install ComputationalHypergraphDiscovery
```


## Quick start

Graph discovery takes very little time. The following code runs the method on the example dataset provided in the repo. The dataset is a 2D array of shape (n_samples, n_features) where each row is a sample and each column is a feature. After fitting the model, the graph is stored in the `GraphDiscovery` object, specifically its graph `G` attribute. The graph is a `networkx` object, which can be easily plotted using `.plot_graph()`.

>You can find the Sachs dataset in the repo, at this [link](https://github.com/TheoBourdais/ComputationalHypergraphDiscovery/blob/main/examples/SachsData.csv).

```python
import ComputationalHypergraphDiscovery as CHD
import pandas as pd
df=pd.read_csv('https://raw.githubusercontent.com/TheoBourdais/ComputationalHypergraphDiscovery/main/examples/SachsData.csv')
df=df.sample(n=500,random_state=1) #subsample to run example quickly
kernels=CHD.Modes.LinearMode()+CHD.Modes.QuadraticMode()
graph_discovery = CHD.GraphDiscovery.from_dataframe(df,mode_kernels=kernels)
graph_discovery.fit()
graph_discovery.plot_graph()
```

## Available modifications of the base algorithm

The code gives an easy-to-use interface to manipulate the graph discovery method. It is designed to be modular and flexible. The main changes you can make are
- **Kernels and modes**: You can decide what type of function will be used to link the nodes. The code provides a set of kernels, but you can easily add your own. The interface is designed to resemble the scikit-learn API, and you can use any kernel from scikit-learn. 
- **Decision logics**: In order to identify the edges of the graph, we need to decide whether certain connections are significant. The code provides indicators (like the level of noise), and the user specifies how to interpret them. The code provides a set of decision logic, but you can define your own. 
- **Clustering**: If a set of nodes is highly dependent, it is possible to merge them into a cluster of nodes. This gives greater readability and prevents the graph discovery method from missing other connections. 
- **Possible edges**: If you know that specific nodes cannot be connected, you can specify it to the algorithm. By default, all edges are possible. 


Full documentation is available [here](https://github.com/TheoBourdais/ComputationalHypergraphDiscovery). 

## Acknowledgements

Copyright 2023 by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

<img style="width:100%;" alt="CHD logo" src="_images/logo.jpg"></a>


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
- add sachs data to repo, and modify code in readme and blog post to use it
- modify gene code to fit new interface
- modify plot code to see individual activations

This is the source code for the paper ["Computational Hypergraph Discovery: A Gaussian process framework for connecting the dots"][paper_url]. 

Please see the [companion blog post](https://theobourdais.github.io/) for a gentle introduction to the method and the code.


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
kernel=CHD.Modes.LinearKernel()+CHD.Modes.QuadraticKernel()
graph_discovery = CHD.GraphDiscovery.from_dataframe(df,mode_kernel=kernel)
graph_discovery.fit()
graph_discovery.plot_graph()
```

You should obtain the following graph:

<img style="width:100%;" alt="Resulting graph Sachs example" src="examples\sachs.png"></a>


## Available modifications of the base algorithm

The code gives an easy-to-use interface to manipulate the graph discovery method. It is designed to be modular and flexible. The main changes you can make are
- **Kernels and modes**: You can decide what type of function will be used to link the nodes. The code provides a set of kernels, but you can easily add your own. The interface is designed to ressemble the scikit-learn API, and you can use any kernel from scikit-learn. 
- **Decision logics**: In order to identify the edges of the graph, we need to decide wether certain connections are significant. The code provides indicators (like level of noise) and the user specifies how to interpret them. The code provides a set of decision logics, but you can define your own. 
- **Clustering**: If a set of nodes is highly dependent, it is possible to merge them into a cluster of nodes. This gives greater readability and prevents the graph discovery method from missing other connections. 


## The Base methods


## Manipulating kernels

## Manipulating decision logics

## Manipulating clusters

<!-- links -->

[paper_url]: https://example.com

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
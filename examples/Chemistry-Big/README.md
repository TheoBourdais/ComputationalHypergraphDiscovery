# Big chemistry example

**Warning**: To properly reproduce the results of the paper, you must switch to the [BCR example branch](https://github.com/TheoBourdais/ComputationalHypergraphDiscovery/tree/BCR-example).

In this example, we tackle a large scale example with 1122 variables and their derivatives. The data is generated using the BCR benchmark from the [Catalyst repository](https://github.com/SciML/Catalyst_PLOS_COMPBIO_2023). The folder is organized as follows:
- `data_generation`: files used for the data generation process
- `data_and_results`: data used for the CHD process, and resulting graphs stored
- `large_experiment_files`: files used to perform the CHD. Because of the large scale, the task was divided in several processes on different GPUs
- `process big chemical reaction.ipynb`: process the results of the first experiment, as described in the paper
- `process big chemical reaction uniform.ipynb`: process the results of the second experiment, as described in the paper. 

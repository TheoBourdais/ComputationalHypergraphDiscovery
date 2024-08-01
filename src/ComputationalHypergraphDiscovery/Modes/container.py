import jax.numpy as jnp


class ModeContainer:
    """
    A class representing a container for modes.

    Attributes:
    - names (list): A list of variable names.
    - name_to_index (dict): A dictionary mapping variable names to their indices.
    - clusters (list): A list of clusters of variable names.
    - index_matrix (numpy.ndarray): A mode matrix.

    Methods:
    - __init__(self, variable_names, clusters=None): Initializes the ModeContainer object.
    - assign_clusters(self, clusters): Assigns clusters to the container.
    - make_index_matrix(names, clusters, name_to_index): Creates an index matrix from the given names and clusters.
    - __repr__(self): Returns a string representation of the object.
    """

    def __init__(
        self,
        variable_names,
        clusters=None,
    ) -> None:
        """
        Initialize the ModeContainer object.

        Args:
        - variable_names (list): A list of variable names.
        - clusters (list, optional): A partition of the variable names. Defaults to None.

        Returns:
        - None
        """
        assert len(list(variable_names)) == len(set(list(variable_names)))
        self.names = variable_names
        self.name_to_index = {name: i for i, name in enumerate(self.names)}
        self.assign_clusters(clusters)
        self.index_matrix = ModeContainer.make_index_matrix(
            self.names, self.clusters, self.name_to_index
        )

    def assign_clusters(self, clusters):
        """
        Assigns clusters to the container. If clusters is None, the clusters are set to be the singleton clusters of the variables.
        Once cluster are assigned, nodes inside of the cluster have to be either all active or all inactive.

        Args:
        - clusters (list): A partition of the array matrices_names.

        Returns:
        - None

        Raises:
        - AssertionError: If the clusters are not a partition of names.
        """
        if clusters is None:
            self.clusters = [[name] for name in self.names]

        else:
            clusters_list = [name for cluster in clusters for name in cluster]
            assert len(clusters_list) == len(set(clusters_list))
            assert len(clusters_list) == len(self.names)
            assert set(clusters_list).issubset(set(list(self.names)))
            self.clusters = clusters

    def make_index_matrix(names, clusters, name_to_index):
        """
        Creates an index matrix from the given names and clusters.

        Args:
        - names (list): A list of variable names.
        - clusters (list): A list of clusters of variable names.
        - name_to_index (dict): A dictionary mapping variable names to their indices.

        Returns:
        - numpy.ndarray: A mode matrix.
        """
        res = jnp.zeros((len(clusters), len(names)))

        # Prepare lists to accumulate indices for batch update
        cluster_indices = []
        name_indices = []

        # Collect indices that need to be set to 1
        for i, cluster in enumerate(clusters):
            for name in cluster:
                if name in name_to_index:  # Check if name is valid
                    cluster_indices.append(i)
                    name_indices.append(name_to_index[name])

        # Use advanced indexing to set multiple indices to 1 at once
        res = res.at[cluster_indices, name_indices].set(1)

        return res

    def __repr__(self) -> str:
        """
        Returns a string representation of the object.
        The string representation is a list of the active clusters in the container.

        Returns:
            str: A string representation of the object.
        """
        res = ["/".join(cluster) for cluster in self.clusters]
        return res.__repr__()

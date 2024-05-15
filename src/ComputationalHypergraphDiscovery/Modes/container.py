import jax.numpy as jnp
import numpy as onp


class ModeContainer:
    """
    A container for mode matrices and their metadata. Allows manipulation of the modes, i.e. removing nodes, handling clusters, etc.

    All modifications of the container are done by creating a new container with the desired modifications. This is done to avoid side effects.
    The matrices are never modified. Instead, the container keeps track of which matrices are used and which are not. Thus, one must view
    the active modes as the modes of the container, while the inactive modes can be viewed as a cache of the deleted modes.

    Parameters
    ----------
    matrices : list of numpy.ndarray
        List of mode matrices.
    matrices_types : list of str
        List of mode types.
    matrices_names : list of str
        List of mode names.
    interpolatory_list : list of bool
        List of boolean values indicating whether each mode is interpolatory.
    variable_names : list of str
        List of variable names.
    beta : numpy.ndarray
        Array of beta scale factors.
    clusters : list of list of str, optional
        List of clusters of variable names. Defaults to None.
    level : numpy.ndarray, optional
        Array of levels for each mode. Defaults to None.
    used : dict of str:bool, optional
        Dictionary of variable names and their usage status. Defaults to None.
    """

    def __init__(
        self,
        variable_names,
        clusters=None,
    ) -> None:
        """
        Initialize the ModeContainer object.
        """
        assert len(list(variable_names)) == len(set(list(variable_names)))
        self.names = variable_names
        self.name_to_index = {name: i for i, name in enumerate(self.names)}
        self.assign_clusters(clusters)
        self.index_matrix = ModeContainer.make_index_matrix(
            self.names, self.clusters, self.name_to_index
        )

    @property
    def node_number(self):
        """
        Returns the number of active clusters in the container (or nodes if there are no clusters).
        """
        return len(self.active_clusters)

    @property
    def active_names(self):
        """
        Returns the names of the active items in the container.
        """
        return self.names[self.used_list]

    def active_clusters(self, active_modes):
        """
        Returns a list of active clusters in the container.
        """
        active_clusters = jnp.sum(active_modes[None, :] * self.index_matrix, axis=1) > 0
        return [
            cluster for cluster, active in zip(self.clusters, active_clusters) if active
        ]

    def assign_clusters(self, clusters):
        """
        Assigns clusters to the container. If clusters is None, the clusters are set to be the singleton clusters of the variables.
        Once cluster are assigned, nodes inside of the cluster have to be either all active or all inactive.

        Args:
        - clusters (list): A partition of the array matrices_names.

        Returns:
        - None

        Raises:
        - AssertionError: If the clusters are not a partition of matrices_names.
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

    def get_index_of_name(self, target_name):
        """
        Returns the index of the mode with the given name.

        Args:
        - target_name (str): The name of the mode to search for.

        Returns:
        - int: The index of the mode with the given name.

        Raises:
        - Exception: If the target name is not in the list of active names.
        - Exception: If the target name is not in the list of names.
        """
        return self.name_to_index[target_name]

    def get_cluster_of_node_name(self, target_name):
        """
        Returns the cluster that contains the given node name.

        Args:
        - target_name (str): The name of the node to search for.

        Returns:
        - list: The cluster that contains the given node name.

        Raises:
        - Exception: If the node name is not found in any of the clusters.
        """
        for i, cluster in enumerate(self.clusters):
            if target_name in cluster:
                return i, cluster
        raise Exception(f"{target_name} was not in the modes' clusters {self.clusters}")

    def get_cluster_by_name(self, cluster_name):
        """
        Returns the cluster with the given name. This method doesn't check for permutations of the cluster, so the name must
        state the cluster in the same order as it was assigned.

        Args:
        - cluster_name (str): The name of the cluster to retrieve.

        Returns:
        - list: The cluster with the given name.

        Raises:
        - Exception: If the cluster with the given name is not found.
        """
        for i, cluster in enumerate(self.clusters):
            if cluster_name == "/".join(cluster):
                return i, cluster
        raise Exception(
            f"{cluster_name} was not in the modes' clusters {self.clusters}"
        )

    def delete_node_by_name(self, target_name, active_modes):
        """
        Deletes the node with the given name from the container.

        Args:
        - target_name (str): The name of the node to be deleted.

        Returns:
        - ModeContainer: A new container with the node deleted.
        """
        return self.delete_cluster(
            self.get_cluster_of_node_name(target_name)[0], active_modes
        )

    def delete_cluster(self, cluster_index, active_modes):
        """
        Deletes the specified cluster from the mode container by setting the 'used' flag to False for all variables in the cluster.

        Args:
        - cluster (list): A list of variable names representing the cluster to be deleted.

        Returns:
        - ModeContainer: A new mode container with the specified cluster deleted.
        """
        return active_modes.at[cluster_index, :].set(0)

    def __repr__(self) -> str:
        """
        Returns a string representation of the object.
        The string representation is a list of the active clusters in the container.

        Returns:
            str: A string representation of the object.
        """
        res = ["/".join(cluster) for cluster in self.clusters]
        return res.__repr__()

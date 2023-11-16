import numpy as np
from .kernels import ModeKernelList


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
        matrices,
        matrices_types,
        matrices_names,
        interpolatory_list,
        variable_names,
        beta,
        clusters=None,
        level=None,
        used=None,
    ) -> None:
        """
        Initialize the ModeContainer object.
        """
        self.constant_mat = np.ones(matrices[0].shape[-2:])
        self.matrices = matrices
        self.matrices_types = matrices_types
        assert len(list(variable_names)) == len(set(list(variable_names)))
        self.names = variable_names
        self.name_to_index = {name: i for i, name in enumerate(self.names)}
        self.beta = beta
        self.level = level
        self.matrices_names = matrices_names
        self.interpolatory_list = interpolatory_list
        self.assign_clusters(clusters)
        if level is None:
            self.level = np.ones_like(matrices_names)
        else:
            self.level = level
        if used is not None:
            self.used = used
        else:
            self.used = {name: True for name in self.names}
        self.used_list = np.array([self.used[name] for name in self.names])

    def from_mode_kernels(X, variable_names, mode_kernels, clusters=None):
        """
        Construct a ModeContainer object from a list of ModeKernels.

        Args:
        - X (np.ndarray): The data tensor.
        - variable_names (list): A list of variable names.
        - mode_kernels (ModeKernelList or ModeKernel): A list of ModeKernels or a single ModeKernel.
        - clusters (list, optional): A list of cluster indices for each variable. Defaults to None.

        Returns:
        - ModeContainer: A container object for the mode matrices.

        Raises:
        - Exception: If mode_kernels is not a ModeKernelList or a ModeKernel.
        """
        try:
            modekernels = ModeKernelList(mode_kernels)
        except AssertionError:
            raise Exception("mode_kernels must be a ModeKernelList or a ModeKernel")
        matrices = modekernels(X)
        matrices_types = [mode.mode_type for mode in modekernels.modes]
        matrices_names = [mode.name for mode in modekernels.modes]
        interpolatory_list = [mode.is_interpolatory for mode in modekernels.modes]
        beta = [mode.beta_scale for mode in modekernels.modes]
        return ModeContainer(
            matrices=matrices,
            matrices_types=matrices_types,
            matrices_names=matrices_names,
            interpolatory_list=interpolatory_list,
            variable_names=variable_names,
            beta=np.array(beta),
            clusters=clusters,
            level=None,
            used=None,
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

    @property
    def active_clusters(self):
        """
        Returns a list of active clusters in the container.
        """
        res = []
        for cluster in self.clusters:
            if self.cluster_is_active(cluster):
                res.append(cluster)
        return res

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

    def cluster_is_active(self, cluster):
        """
        Check if all elements in the given cluster are either active or inactive at the same time.

        Args:
        - cluster (list): A list of names of variables.

        Returns:
        - bool: True if all elements in the cluster are active, False if all elements are inactive.

        Raises:
        - AssertionError: If some elements in the cluster are active while others are inactive.
        - KeyError: If some elements in the cluster are not in the container.
        """
        cluster_usage = [self.used[name] for name in cluster]
        assert (
            len(set(cluster_usage)) == 1
        ), f'all elements in cluster {"/".join(cluster)} must be either active or inactive at the same time'
        return cluster_usage[0]

    def is_interpolatory(self, chosen_level=None):
        """
        Check if the container is interpolatory at a given level.

        Args:
        - chosen_level (int, optional): The level to check. Defaults to None.
            In that case the level is the current level of the container, a value stored in self.level.

        Returns:
        - bool: True if the container is interpolatory at the given level, False otherwise.
        """
        level = self.get_level(chosen_level)
        res = False
        for li, is_interpolatory_bool in zip(level, self.interpolatory_list):
            res = res or (
                is_interpolatory_bool and li == 1
            )  # if one mode is interpolatory and active, the container is interpolatory
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
        try:
            assert self.used[target_name]
        except KeyError or AssertionError:
            raise Exception(
                f"{target_name} is not in the modes' list of active names {self}"
            )
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
        for cluster in self.clusters:
            if target_name in cluster:
                return cluster
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
        for cluster in self.clusters:
            if cluster_name == "/".join(cluster):
                return cluster
        raise Exception(
            f"{cluster_name} was not in the modes' clusters {self.clusters}"
        )

    def delete_node(self, index):
        """
        Deletes a node from the container by its index.

        Args:
        - index (int): The index of the node to be deleted.

        Returns:
        - ModeContainer: A new container with the node deleted.
        """
        return self.delete_node_by_name(self.names[index])

    def delete_node_by_name(self, target_name):
        """
        Deletes the node with the given name from the container.

        Args:
        - target_name (str): The name of the node to be deleted.

        Returns:
        - ModeContainer: A new container with the node deleted.
        """
        return self.delete_cluster(self.get_cluster_of_node_name(target_name))

    def delete_cluster_by_name(self, cluster_name):
        """
        Deletes a cluster by its name.

        Args:
        - cluster_name (str): The name of the cluster to delete.

        Returns:
        - ModeContainer: A new container with the cluster deleted.
        """
        return self.delete_cluster(self.get_cluster_by_name(cluster_name))

    def delete_cluster(self, cluster):
        """
        Deletes the specified cluster from the mode container by setting the 'used' flag to False for all variables in the cluster.

        Args:
        - cluster (list): A list of variable names representing the cluster to be deleted.

        Returns:
        - ModeContainer: A new mode container with the specified cluster deleted.
        """
        new_used = self.used.copy()
        for name in cluster:
            assert self.used[name]
            new_used[name] = False
        return ModeContainer(
            matrices=self.matrices,
            matrices_types=self.matrices_types,
            matrices_names=self.matrices_names,
            interpolatory_list=self.interpolatory_list,
            variable_names=self.names,
            beta=self.beta,
            level=self.level,
            used=new_used,
            clusters=self.clusters,
        )

    def get_level(self, chosen_level):
        """
        Returns the level of the container object. The level is an array of 1 and 0 indicating which kernels are active and which are not.

        If chosen_level is None, returns the current level of the container object.
        If chosen_level is not None, returns the level corresponding to the given level name.

        Args:
        - chosen_level (str or None): The name of the level to return. If None, returns the current level.

        Returns:
        - numpy.ndarray: The level of the container object.
        """
        if chosen_level is None:
            return self.level
        level = []
        found = False
        for level_name in self.matrices_names:
            if not found:
                level.append(1)
            else:
                level.append(0)
            if level_name == chosen_level:
                found = True
        if not found:
            raise Exception(
                f"Level {chosen_level} is not in the list of levels {self.matrices_names}"
            )
        return np.array(level)

    def set_level(self, chosen_level):
        """
        Set the level of the container to the chosen level.

        Args:
        - chosen_level (str): The level to set the container to.

        Returns:
        - None
        """
        assert chosen_level is not None
        self.level = self.get_level(chosen_level)

    def prod_and_sum(matrix, mask):
        """
        Computes the product and sum of a matrix along the first axis, where the sum is taken over the elements of the matrix
        that correspond to a True value in the mask array. This allows to compute the combinatorial kernels, and the mask allows to ignore the
        inactive values

        Args:
        - matrix (numpy.ndarray): The input matrix.
        - mask (numpy.ndarray): A boolean array of the same shape as the first axis of the matrix. The i-th element of the mask
            indicates whether the i-th kernel is active in the computation.

        Returns:
        - numpy.ndarray: The resulting array of shape (matrix.shape[1], matrix.shape[2]), where the i,j-th element is
            the product of the i,j-th elements of the matrix that correspond to a True value in the mask array, summed over
            the first axis of the matrix.
        """
        return np.prod(
            np.add(matrix, np.ones_like(matrix), where=mask[:, None, None]),
            axis=0,
            where=mask[:, None, None],
        )

    def sum_a_matrix(matrix, matrix_type, used):
        """
        Sums the elements of a matrix based on the specified matrix type and which kernels are active (using the mask).

        Args:
        - matrix (numpy.ndarray): The matrix to sum.
        - matrix_type (str): The type of matrix. Must be one of "individual", "pairwise", or "combinatorial".
        - used (numpy.ndarray): A boolean mask indicating which elements of the matrix to include in the sum.

        Returns:
        - numpy.ndarray: The sum of the specified elements of the matrix.
        """
        if matrix_type == "individual":
            return np.sum(matrix, axis=0, where=used[:, None, None])
        if matrix_type == "pairwise":
            used_2D = used[:, None] * used[None, :]
            return np.sum(matrix, axis=(0, 1), where=used_2D[:, :, None, None])
        if matrix_type == "combinatorial":
            return ModeContainer.prod_and_sum(matrix, used)
        raise f"Unknown matrix type {matrix_type}"

    def sum_a_matrix_of_index(matrix, matrix_type, used, index):
        """
        Sums a matrix of a given type over a specified index. This allows to compute the matrix associated with a specific node.
        The mask is used to ignore the inactive values and only sum with the chosen index

        Args:
        - matrix (numpy.ndarray): The matrix to sum over.
        - matrix_type (str): The type of matrix. Can be "individual", "pairwise", or "combinatorial".
        - used (numpy.ndarray): A boolean array indicating which indices have already been used.
        - index (int): The index to sum over.

        Returns:
        - numpy.ndarray: The sum of the matrix over the specified index.
        """
        if matrix_type == "individual":
            return matrix[index]
        if matrix_type == "pairwise":
            mask = np.zeros(matrix.shape[0], dtype=bool)
            mask[index] = True
            used_2D = mask[:, None] * used[None, :] + used[:, None] * mask[None, :]
            return np.sum(matrix, axis=(0, 1), where=used_2D[:, :, None, None])
        if matrix_type == "combinatorial":
            used_for_prod = used.copy()
            used_for_prod[index] = False
            return matrix[index] * ModeContainer.prod_and_sum(matrix, used_for_prod)
        raise f"Unknown matrix type {matrix_type}"

    def sum_a_matrix_of_indexes(matrix, matrix_type, used, indexes):
        """
        Sums the values of a matrix along the specified indexes. This allows to compute the matrix associated with a specific cluster.
        The mask is used to ignore the inactive values and only sum with the chosen cluster.

        Args:
        - matrix (numpy.ndarray): The matrix to sum.
        - matrix_type (str): The type of matrix. Must be one of "individual", "pairwise", or "combinatorial".
        - used (numpy.ndarray): A boolean array indicating which indices have already been used.
        - indexes (numpy.ndarray): The indices to sum along.

        Returns:
        - numpy.ndarray: The sum of the matrix along the specified indices.
        """
        mask = np.zeros(matrix.shape[0], dtype=bool)
        mask[indexes] = True
        if matrix_type == "individual":
            return np.sum(matrix, axis=0, where=mask[:, None, None])
        if matrix_type == "pairwise":
            used_2D = mask[:, None] * used[None, :] + used[:, None] * mask[None, :]
            return np.sum(matrix, axis=(0, 1), where=used_2D[:, :, None, None])
        if matrix_type == "combinatorial":
            used_for_prod = used.copy()
            used_for_prod[indexes] = False
            return (
                ModeContainer.prod_and_sum(matrix, mask) - 1
            ) * ModeContainer.prod_and_sum(matrix, used_for_prod)
        raise f"Unknown matrix type {matrix_type}"

    def get_K(self, chosen_level=None):
        """
        Computes the kernel matrix K of the ModeContainer, at the chosen level.

        Args:
        - chosen_level (int, optional): The level for which to compute K. If None, the current level is used.

        Returns:
        - numpy.ndarray: The hypergraph Laplacian matrix K.
        """
        level = self.get_level(chosen_level)
        K = np.zeros_like(self.constant_mat)
        K += self.constant_mat
        for i, matrix in enumerate(self.matrices):
            if level[i] != 0:  # level==0 means the mode is inactive
                K += self.beta[i] * ModeContainer.sum_a_matrix(
                    matrix, self.matrices_types[i], self.used_list
                )
        return K

    def get_K_of_name(self, name):
        """
        Returns the matrix kernel K of the mode associated with the chosen name at the current level.

        Args:
        - name (str): The name of the node.

        Returns:
        - numpy.ndarray: The matrix kernel K of the node.
        """
        return self.get_K_of_index(self.get_index_of_name(name))

    def get_K_of_index(self, index):
        """
        Returns the matrix kernel K of the mode associated with the chosen index at the current level.

        Args:
            index (int): The index of the node.

        Returns:
            numpy.ndarray: The matrix kernel K of the node at given index.
        """
        assert self.used_list[index]
        res = np.zeros_like(self.constant_mat)
        for i, matrix in enumerate(self.matrices):
            if self.level[i] != 0:  # level==0 means the mode is inactive
                res += self.beta[i] * ModeContainer.sum_a_matrix_of_index(
                    matrix, self.matrices_types[i], self.used_list, index
                )
        return res

    def get_K_of_cluster(self, cluster):
        """
        Returns the matrix kernel K of the mode associated with the chosen cluster at the current level.

        Args:
        - cluster (list): A list of node names in the cluster.

        Returns:
        - numpy.ndarray: The matrix kernel K of the cluster.
        """
        if len(cluster) == 1:
            return self.get_K_of_name(cluster[0])
        assert self.cluster_is_active(cluster)
        indexes = [self.get_index_of_name(name) for name in cluster]
        res = np.zeros_like(self.constant_mat)
        for i, matrix in enumerate(self.matrices):
            if self.level[i] != 0:  # level==0 means the mode is inactive
                res += self.beta[i] * ModeContainer.sum_a_matrix_of_indexes(
                    matrix, self.matrices_types[i], self.used_list, indexes
                )
        return res

    def get_K_without_index(self, index):
        """
        Returns the matrix kernel K of the mode after having removed the node at the given index at the current level.

        Args:
        - index (int): index of the node to remove

        Returns:
        - K (np.ndarray): The matrix kernel K of the container after removing node at index.
        """
        assert self.used_list[index]
        return self.delete_node(index).get_K()

    def get_K_without_name(self, name):
        """
        Returns the matrix kernel K of the mode after having removed the node at the given name at the current level.

        Args:
        - name (str): name of the node to remove

        Returns:
        - K (np.ndarray): The matrix kernel K of the container after removing node.
        """
        assert self.used[name]
        return self.delete_node_by_name(name).get_K()

    def get_K_without_cluster(self, cluster):
        """
        Returns the matrix kernel K of the mode after having removed the given cluster at the current level.

        Args:
        - cluster (list): A list of node names in the cluster.

        Returns:
        - K (np.ndarray): The matrix kernel K of the container after removing the cluster.
        """
        assert self.cluster_is_active(cluster)
        return self.delete_cluster(cluster).get_K()

    def __repr__(self) -> str:
        """
        Returns a string representation of the object.
        The string representation is a list of the active clusters in the container.

        Returns:
            str: A string representation of the object.
        """
        res = ["/".join(cluster) for cluster in self.active_clusters]
        return res.__repr__()

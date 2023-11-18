import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def partition_layout(g, partition, ratio=0.3):
    """
    Compute the layout for a modular graph.

    Args:
    - g (networkx.Graph or networkx.DiGraph): the graph to plot
    - partition (dict): A dictionary mapping node IDs to group IDs.
    - ratio (float, optional): Controls how tightly the nodes are clustered around their partition centroid.
        If 0, all nodes of a partition are at the centroid position.
        If 1, nodes are positioned independently of their partition centroid.

    Returns:
    - pos (dict): A dictionary mapping node IDs to their positions.
    - hypergraph (dict): A dictionary representing the hypergraph.


    """

    pos_nodes_communities, pos_communities = _position_communities(g, partition)
    assert set(pos_nodes_communities.keys()).isdisjoint(
        set(pos_communities.keys())
    ), "node names and community names must be different"

    pos_nodes = _position_nodes(g, partition)
    pos_nodes = {k: ratio * v for k, v in pos_nodes.items()}

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_nodes_communities[node] + pos_nodes[node]
    pos.update(pos_communities)

    hypergraph = _make_hypergraph(g, partition)

    return pos, hypergraph


def _make_hypergraph(g, partition):
    """
    Creates a hypergraph from a graph and a partition.

    Parameters:
    - g (nx.Graph): The input graph.
    - partition (dict): A dictionary mapping nodes to their respective communities.

    Returns:
    - nx.DiGraph: The resulting hypergraph.
    """
    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    hypergraph.add_nodes_from(g.nodes())

    for ni, nj, data_dict in g.edges(data=True):
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            hypergraph.add_edge(ci, nj, **data_dict, intra_cluster=False)
        else:
            hypergraph.add_edge(ni, nj, **data_dict, intra_cluster=True)
    return hypergraph


def _position_communities(g, partition, **kwargs):
    """
    Compute the positions of nodes in a graph based on the communities they belong to.

    Parameters:
    - g (networkx.Graph): The input graph.
    - partition (dict): A dictionary mapping node IDs to community IDs.
    - **kwargs: Additional keyword arguments to pass to nx.circular_layout.

    Returns:
    - pos (dict): A dictionary mapping node IDs to their positions in the layout.
    - pos_communities (dict): A dictionary mapping community IDs to their positions in the layout.
    """
    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.circular_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos, pos_communities


def _find_between_community_edges(g, partition):
    """
    Given a graph and a partition of its nodes, returns a dictionary of edges connecting different communities.

    Args:
    - g: networkx.Graph object
    - partition: dictionary mapping node IDs to community IDs

    Returns:
    - edges: dictionary mapping pairs of community IDs to lists of edges connecting them
    """
    edges = dict()

    for ni, nj in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges


def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.

    Parameters:
    - g (networkx.Graph): The graph to position nodes in.
    - partition (dict): A dictionary mapping nodes to their respective communities.
    - **kwargs: Additional keyword arguments to pass to the circular_layout function.

    Returns:
    - dict: A dictionary mapping nodes to their positions.
    """
    communities = dict()
    for node, community in partition.items():
        if community in communities:
            communities[community] += [node]
        else:
            communities[community] = [node]

    pos = dict()
    for community, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.circular_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos


def plot_noise_evolution(ancestor_number, list_of_noises, list_of_Zs, ancestor_modes):
    """
    Plots the evolution of noise and noise increment over the number of ancestors.

    Parameters:
    - ancestor_number (list): A list of integers representing the number of ancestors.
    - list_of_noises (list): A list of floats representing the noise.
    - list_of_Zs (list): A list of tuples representing the 5% and 95% quantiles of random noise.
    - ancestor_modes (None or object): An object representing the chosen number of ancestors.

    Returns:
    - fig (matplotlib.figure.Figure): A matplotlib figure object.
    - axes (numpy.ndarray): A numpy array of matplotlib axes objects.
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    axes[0].plot(ancestor_number, list_of_noises, label="noise")
    axes[0].plot(
        ancestor_number,
        [z[0] for z in list_of_Zs],
        label="5% quantile of random noise",
    )
    axes[0].plot(
        ancestor_number,
        [z[1] for z in list_of_Zs],
        label="95% quantile of random noise",
    )
    # color in between the two lines above
    axes[0].fill_between(
        ancestor_number,
        [z[0] for z in list_of_Zs],
        [z[1] for z in list_of_Zs],
        alpha=0.2,
    )

    axes[1].plot(
        ancestor_number,
        [
            list_of_noises[i + 1] - list_of_noises[i]
            for i in range(len(list_of_noises) - 1)
        ]
        + [1 - list_of_noises[-1]],
        label="noise increment",
    )
    if ancestor_modes is not None:
        axes[0].axvline(
            x=ancestor_modes.node_number,
            linestyle="--",
            color="k",
            label=f"chosen number of ancestors={ancestor_modes.node_number}",
        )
        axes[1].axvline(
            x=ancestor_modes.node_number,
            linestyle="--",
            color="k",
            label=f"chosen number of ancestors={ancestor_modes.node_number}",
        )
    axes[0].legend()
    axes[0].set_xlabel("number of ancestors")
    axes[0].set_ylabel("noise")
    axes[0].invert_xaxis()
    axes[0].set_xticks(
        np.linspace(len(list_of_noises), 1, 6, dtype=np.int32, endpoint=True)
    )
    axes[1].legend()
    axes[1].set_xlabel("number of ancestors")
    axes[1].set_ylabel("noise increment")
    axes[1].invert_xaxis()
    axes[1].set_xticks(
        np.linspace(len(list_of_noises), 1, 6, dtype=np.int32, endpoint=True)
    )
    fig.tight_layout()
    return fig, axes

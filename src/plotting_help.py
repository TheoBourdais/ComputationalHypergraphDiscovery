import numpy as onp
import matplotlib.pyplot as plt
import networkx as nx


NODE_LAYOUT = nx.circular_layout
COMMUNITY_LAYOUT = nx.circular_layout


def partition_layout(g, partition, ratio=0.3):
    """
    Compute the layout for a modular graph.

    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        network to plot

    partition -- dict mapping node -> community or None
        Network partition, i.e. a mapping from node ID to a group ID.

    ratio: 0 < float < 1.
        Controls how tightly the nodes are clustered around their partition centroid.
        If 0, all nodes of a partition are at the centroid position.
        if 1, nodes are positioned independently of their partition centroid.

    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions

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
    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = COMMUNITY_LAYOUT(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos, pos_communities


def _find_between_community_edges(g, partition):
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
        pos_subgraph = NODE_LAYOUT(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos


def _layout(networkx_graph):
    edge_list = [edge for edge in networkx_graph.edges]
    node_list = [node for node in networkx_graph.nodes]

    pos = nx.circular_layout(edge_list)

    # NB: some nodes might not be connected and hence will not be in the edge list.
    # Assuming a [0, 0, 1, 1] canvas, we assign random positions on the periphery
    # of the existing node positions.
    # We define the periphery as the region outside the circle that covers all
    # existing node positions.
    xy = list(pos.values())
    centroid = onp.mean(xy, axis=0)
    delta = xy - centroid[onp.newaxis, :]
    distance = onp.sqrt(onp.sum(delta**2, axis=1))
    radius = onp.max(distance)

    connected_nodes = set(_flatten(edge_list))
    for node in node_list:
        if not (node in connected_nodes):
            pos[node] = _get_random_point_on_a_circle(centroid, radius)

    return pos


def _flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]


def _get_random_point_on_a_circle(origin, radius):
    x0, y0 = origin
    random_angle = 2 * onp.pi * onp.random.random()
    x = x0 + radius * onp.cos(random_angle)
    y = y0 + radius * onp.sin(random_angle)
    return onp.array([x, y])


def test():
    # create test data
    cliques = 8
    clique_size = 7
    g = nx.connected_caveman_graph(cliques, clique_size)
    partition = {ii: onp.int(ii / clique_size) for ii in range(cliques * clique_size)}

    pos = partition_layout(g, partition, ratio=0.2)
    nx.draw(g, pos, node_color=list(partition.values()))
    plt.show()


def test2():
    # create test data
    cliques = 8
    clique_size = 7
    g = nx.connected_caveman_graph(cliques, clique_size)
    partition = {ii: onp.int(ii / clique_size) for ii in range(cliques * clique_size)}

    # add additional between-clique edges
    total_nodes = cliques * clique_size
    for ii in range(cliques):
        start = ii * clique_size + int(clique_size / 2)
        stop = (ii + cliques / 2) * clique_size % total_nodes + int(clique_size / 2)
        g.add_edge(start, stop)

    pos = partition_layout(g, partition, ratio=0.2)
    nx.draw(g, pos, node_color=list(partition.values()))
    plt.show()


def plot_noise_evolution(ancestor_number, list_of_noises, list_of_Zs, ancestor_modes):
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
        onp.linspace(len(list_of_noises), 1, 6, dtype=onp.int32, endpoint=True)
    )
    axes[1].legend()
    axes[1].set_xlabel("number of ancestors")
    axes[1].set_ylabel("noise increment")
    axes[1].invert_xaxis()
    axes[1].set_xticks(
        onp.linspace(len(list_of_noises), 1, 6, dtype=onp.int32, endpoint=True)
    )
    fig.tight_layout()
    return fig, axes


if __name__ == "__main__":
    test()
    test2()

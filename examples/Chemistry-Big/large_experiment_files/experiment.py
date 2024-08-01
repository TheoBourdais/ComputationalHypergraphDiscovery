import argparse


def main():
    parser = argparse.ArgumentParser(description="Launch an experiment")
    parser.add_argument("--device", type=int, help="Device ID to use")
    parser.add_argument("--run_index", type=int, help="Index of the run")

    args = parser.parse_args()
    print(f"Using device ID: {args.device}")
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
    os.environ["JAX_TRACEBACK_FILTERING"] = "off"

    # preallocate 95% of GPU memory
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

    import sys

    sys.path.append("../../..")

    import jax
    import pickle
    import pandas as pd
    from src import ComputationalHypergraphDiscovery as CHD

    # from ComputationalHypergraphDiscovery.Modes import LinearMode, QuadraticMode, GaussianMode
    import numpy as onp
    import matplotlib.pyplot as plt
    import networkx as nx

    targets = list(pd.read_csv("batches.csv").iloc[args.run_index])
    targets = [t for t in targets if t == t]

    df = pd.read_csv("../data_and_results/BCR_uniform.csv")
    df = df[df.columns[df.std(axis=0) != 0]]
    df = df[list(df.columns[:1122]) + targets]
    cut = 600
    df_train = df[:cut]
    possible_edges = nx.DiGraph()
    edges = []
    for f_node in df.columns:
        if "partial" in f_node:
            continue
        for nf_node in df.columns:
            if "partial" in nf_node:
                edges.append((f_node, nf_node))
                edges.append((nf_node, f_node))
    possible_edges.add_edges_from(edges)

    graph_discovery = CHD.GraphDiscovery.from_dataframe(
        df_train,
        normalize=True,
        possible_edges=possible_edges,
        kernels=[CHD.Modes.QuadraticMode(memory_efficient_required=True)],
        gamma_min=2e-6,
    )
    mode_chooser = CHD.decision.ThresholdModeChooser(threshold=0.025)
    print(f"pruning {targets}")
    graph_discovery.fit(
        targets,
        mode_chooser=mode_chooser,
        message=f"experiment {args.device}_{args.run_index}",
    )
    # save graph_discovery.G with name that uses time and run name
    save_name = f"../data_and_results/results/G_{args.device}_{args.run_index}.pkl"

    pickle.dump(graph_discovery.G, open(save_name, "wb"))


if __name__ == "__main__":

    main()

# snapshot.py
import os
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def draw_network(weights_file="weights.json", step=10, save_dir="images", accuracy=None, weights_input_hidden=None, bias_hidden=None, weights_hidden_output=None, bias_output=None):
    """
    Draws a snapshot of the neural network from saved weights and biases.

    Args:
        weights_file (str): Path to JSON file containing weights/biases
        step (int): Training step number (used for filename)
        save_dir (str): Folder to save output images
    """

    if weights_input_hidden is None or bias_hidden is None or weights_hidden_output is None or bias_output is None:
        # 1. Load weights + biases
        with open(weights_file, "r") as f:
            data = json.load(f)

        weights_input_hidden = np.array(data["W1"])   # hidden x input
        bias_hidden = np.array(data["b1"])            # hidden x 1
        weights_hidden_output = np.array(data["W2"])  # output x hidden
        bias_output = np.array(data["b2"])            # output x 1



    # 2. Build networkx graph
    G = nx.DiGraph()

    # Node labels
    #input_nodes = [f"I{i}" for i in range(len(weights_input_hidden[0]))]
    input_nodes = ["X coord", "Y coord"]
    hidden_nodes = [f"H{j}" for j in range(len(weights_input_hidden))]
    #output_nodes = [f"O{k}" for k in range(len(weights_hidden_output))]
    output_nodes = ["P(Rule)"]


    # Add nodes
    G.add_nodes_from(input_nodes, layer="input")
    G.add_nodes_from(hidden_nodes, layer="hidden")
    G.add_nodes_from(output_nodes, layer="output")

    # Add edges (input → hidden)
    for i, inp in enumerate(input_nodes):
        for j, hid in enumerate(hidden_nodes):
            w = weights_input_hidden[j][i]
            G.add_edge(inp, hid, weight=w)

    # Add edges (hidden → output)
    for j, hid in enumerate(hidden_nodes):
        for k, out in enumerate(output_nodes):
            w = weights_hidden_output[k][j]
            G.add_edge(hid, out, weight=w)

    # 3. Layout positions
    pos = {}
    # Input layer
    for i, node in enumerate(input_nodes):
        pos[node] = (0, -i - 1)
    # Hidden layer
    for j, node in enumerate(hidden_nodes):
        pos[node] = (1, -j)
    # Output layer
    for k, node in enumerate(output_nodes):
        pos[node] = (2, -k - 1.5)

    # 4. Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=input_nodes, node_color="lightblue", node_size=1500)
    nx.draw_networkx_nodes(G, pos, nodelist=hidden_nodes, node_color="lightgreen", node_size=800)
    nx.draw_networkx_nodes(G, pos, nodelist=output_nodes, node_color="salmon", node_size=1500)

    # 5. Draw edges with weight-based colors/thickness
    edges = G.edges(data=True)

    edge_colors = []
    edge_widths = []
    for u, v, d in edges:
        w = d["weight"]
        # Example mapping: red = negative, blue = positive, thickness = magnitude
        color = "red" if w < 0 else "blue"
        width = max(0.5, abs(w) * 2)  # scale thickness
        edge_colors.append(color)
        edge_widths.append(width)

    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color=edge_colors, width=edge_widths, arrows=False)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")

    # 6. Save image
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, f"snapshot_{step:05d}.png")

    plt.axis("off")
    plt.title(f"Neural Net at Step {step}")
    if accuracy is not None:
        plt.suptitle(f"Accuracy: {accuracy*100:.2f}%", y=0.89, fontsize=10)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    #print(f"Saved snapshot: {filename}")


# Example usage (standalone run)
if __name__ == "__main__":
    draw_network(step=100, accuracy=.25)

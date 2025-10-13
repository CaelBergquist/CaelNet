from model import SimpleNN
import os

#model = SimpleNN(input_dim=2, hidden_dim=4, output_dim=1)

#print(model.get_inner_probability())

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import json
import plotly.graph_objects as go


def planar_draw():
    with open("weights.json", "r") as f:
        data = json.load(f)

    W1 = np.array(data["W1"])   # shape (4, 2)
    b1 = np.array(data["b1"]).flatten()  # shape (4,)

    # Grid for inputs
    x_vals = np.linspace(0, 1, 25)
    y_vals = np.linspace(0, 1, 25)

    X, Y = np.meshgrid(x_vals, y_vals)

    fig = go.Figure()

    for j in range(W1.shape[0]):
        w = W1[j]; b = b1[j]
        Z = w[0]*X + w[1]*Y + b
        fig.add_surface(x=X, y=Y, z=Z, opacity=1)

    # Decision plane
    Z_decision = 0.5 * np.ones_like(X)
    fig.add_surface(x=X, y=Y, z=Z_decision, opacity=1, colorscale="gray")

    fig.show()

# Example loss function (replace with your real loss)
def loss_fn(w1, w2, w3, w4):
    return (w1 - 1)**2 + (w2 + 2)**2 + (w3 - 0.5)**2 + (w4 - 1.5)**2

# Fix w3, w4 at current values
def plot():
    w3, w4 = 0.5, -1.5

    # Grid for w1, w2
    w1_vals = np.linspace(-3, 3, 100)
    w2_vals = np.linspace(-3, 3, 100)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    Z = loss_fn(W1, W2, w3, w4)

    # 3D surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(W1, W2, Z, cmap='viridis')
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_zlabel("Loss")
    plt.show()




def prepend_folder(folder, prepend_str):
    for filename in os.listdir(folder):
        new_filename = prepend_str + filename
        os.rename(os.path.join(folder, filename), os.path.join(folder, new_filename))

def delete_first_char(folder):
    for filename in os.listdir(folder):
        new_filename = filename[1:]  # Remove the first character
        os.rename(os.path.join(folder, filename), os.path.join(folder, new_filename))

#prepend_folder("images2", "3")
#delete_first_char("images2")
#plot()
planar_draw()

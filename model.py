# model.py
import numpy as np
import json
import os
from rule import is_inside


class SimpleNN:
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1, weights_file="weights.json"):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.innerProb = self.get_inner_probability()

        # Try to load weights from JSON if the file exists
        if weights_file and os.path.exists(weights_file):
            try:
                with open(weights_file, "r") as f:
                    data = json.load(f)

                # assign and convert to numpy arrays
                self.W1 = np.array(data["W1"], dtype=float)
                self.b1 = np.array(data["b1"], dtype=float)
                self.W2 = np.array(data["W2"], dtype=float)
                self.b2 = np.array(data["b2"], dtype=float)

                # Normalize bias shapes to (N,1) if they were saved as 1D lists
                if self.b1.ndim == 1:
                    self.b1 = self.b1.reshape((self.b1.shape[0], 1))
                if self.b2.ndim == 1:
                    self.b2 = self.b2.reshape((self.b2.shape[0], 1))

                # simple shape checks (optional, will raise if mismatched)
                assert self.W1.shape == (hidden_dim, input_dim)
                assert self.b1.shape == (hidden_dim, 1)
                assert self.W2.shape == (output_dim, hidden_dim)
                assert self.b2.shape == (output_dim, 1)

                return  # loaded successfully, done
            except Exception as e:
                # If anything goes wrong, fall back to random init
                print(f"[model] Warning loading {weights_file}: {e}. Falling back to random init.")

        # Fallback: small random initialization (same as before)
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b1 = np.zeros((hidden_dim, 1))
        self.W2 = np.random.randn(output_dim, hidden_dim) * 0.01
        self.b2 = np.zeros((output_dim, 1))

    def relu(self, z): # ReLU - zero negative values
        return np.maximum(0, z)

    def sigmoid(self, z): # Sigmoid activation for output layer - maps to (0,1)
        return 1 / (1 + np.exp(-z))

    def forward(self, x):
        """
        Forward pass through the network.
        x: input vector of shape (2,) or (2, 1)
        Returns predicted probability (y_hat).
        """
        # Ensure column vector
        x = x.reshape(-1, 1)  

        # Hidden layer
        z1 = np.dot(self.W1, x) + self.b1
        a1 = self.relu(z1)

        # Output layer
        z2 = np.dot(self.W2, a1) + self.b2
        y_hat = self.sigmoid(z2)

        return y_hat
    
    def get_inner_probability(self):
        #returns the probability that a random point is inside the shape
        depth = 700
        total = 0
        for x in range(depth):
            for y in range(depth):
                if is_inside(x/depth, y/depth):
                    total += 1
        return round(total / depth**2, 3) #crop to 3 decimal places
        
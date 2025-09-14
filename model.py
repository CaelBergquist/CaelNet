# model.py
import numpy as np

class SimpleNN:
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1):
        """
        Initialize weights and biases for a 2 -> 4 -> 1 network.
        Weights are small random numbers, biases are zeros.
        """
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b1 = np.zeros((hidden_dim, 1))
        self.W2 = np.random.randn(output_dim, hidden_dim) * 0.01
        self.b2 = np.zeros((output_dim, 1))

        '''
        W1 = (4 rows, 2 columns)
        b1 = (2, 1)
        W2 = (1, 4)
        b2 = (1, 1)
        '''

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
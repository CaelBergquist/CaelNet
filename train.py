# train.py
import numpy as np
from model import SimpleNN
from rule import is_inside

class Trainer:
    def __init__(self, model: SimpleNN, learning_rate=0.01):
        self.model = model
        self.lr = learning_rate

    def binary_cross_entropy(self, y_hat, y):
        """
        Binary cross-entropy loss.
        y_hat: predicted probability (scalar, 1x1 array)
        y: true label (0 or 1)
        """
        eps = 1e-15  # avoid log(0)
        y_hat_clipped = np.clip(y_hat, eps, 1 - eps)
        loss = -(y * np.log(y_hat_clipped) + (1 - y) * np.log(1 - y_hat_clipped))
        return loss

    def forward_and_backward(self, x, y):
        """
        Runs forward pass, computes loss, and performs backpropagation.
        Updates model weights in-place.
        """
        # --- Forward pass ---
        x = x.reshape(-1, 1)  # shape (2,1)

        # Input -> Hidden
        z1 = np.dot(self.model.W1, x) + self.model.b1   # (4,1)
        a1 = self.model.relu(z1)                        # (4,1)

        # Hidden -> Output
        z2 = np.dot(self.model.W2, a1) + self.model.b2  # (1,1)
        y_hat = self.model.sigmoid(z2)                  # (1,1) end of forward pass
        if abs(y - y_hat) < 0.5:
            correct = 1
        else:
            correct = 0

        # --- Compute loss ---
        loss = self.binary_cross_entropy(y_hat, y)

        # --- Backpropagation ---
        # Output layer gradients
        dz2 = y_hat - y                        # (1,1) error at output layer, confidence - truth
        dW2 = np.dot(dz2, a1.T)                # (1,4) 
        db2 = dz2                              # (1,1)

        # Hidden layer gradients
        da1 = np.dot(self.model.W2.T, dz2)     # (4,1)
        dz1 = da1 * (z1 > 0)                   # derivative of ReLU
        dW1 = np.dot(dz1, x.T)                 # (4,2)
        db1 = dz1                              # (4,1)

        # --- Update parameters (gradient descent) ---
        self.model.W1 -= self.lr * dW1
        self.model.b1 -= self.lr * db1
        self.model.W2 -= self.lr * dW2
        self.model.b2 -= self.lr * db2

        return loss, correct

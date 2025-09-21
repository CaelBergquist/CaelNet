import numpy as np
from model import SimpleNN
from train import Trainer
from rule import is_inside
import json  # For saving/loading weights if desired
from main import num_iterations, learning_rate
import os
from snapshot2 import draw_points

#np.random.seed(42)
WEIGHTS_JSON = "weights.json"
TEST_LOG = "test_log.json"
NUM_TEST = 10000

if not os.path.exists(WEIGHTS_JSON):
    raise FileNotFoundError(f"{WEIGHTS_JSON} not found. Create it with your reset script first.")

with open(WEIGHTS_JSON, "r") as f:
    w = json.load(f)

model = SimpleNN(input_dim=2, hidden_dim=4, output_dim=1)

print ("evaluating performance on 1000 random points...")
print ("Random accuracy: 25%")
correct_count = 0

model.W1 = np.array(w["W1"])
model.b1 = np.array(w["b1"])
model.W2 = np.array(w["W2"])
model.b2 = np.array(w["b2"])

assert model.W1.shape == (4, 2), f"W1 shape {model.W1.shape} unexpected"
assert model.b1.shape == (4, 1), f"b1 shape {model.b1.shape} unexpected"
assert model.W2.shape == (1, 4), f"W2 shape {model.W2.shape} unexpected"
assert model.b2.shape == (1, 1), f"b2 shape {model.b2.shape} unexpected"

data = []

for i in range(NUM_TEST):
    x = np.array([np.random.rand(), np.random.rand()])  
    y_true = is_inside(x[0], x[1])
    y_pred = model.forward(x)
    y_pred_label = 1 if y_pred >= 0.5 else 0
    if y_true == y_pred_label:
            # Append new entry
        entry = {"x": float(x[0]), "y": float(x[1]), "correct": "y"}
        data.append(entry)
        correct_count += 1
    else:
        entry = {"x": float(x[0]), "y": float(x[1]), "correct": "n"}
        data.append(entry)

print("Accuracy:", correct_count / NUM_TEST)

with open(TEST_LOG, "w") as f:
    json.dump(data, f, indent=2)

draw_points(step=0)
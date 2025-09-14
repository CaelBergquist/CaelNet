import numpy as np
from model import SimpleNN
from train import Trainer
from rule import is_inside
import json  # For saving/loading weights if desired
from main import num_iterations, learning_rate
import os


with open('test_log.json', "r") as f:
        data = json.load(f)


model = SimpleNN(input_dim=2, hidden_dim=4, output_dim=1)

print ("evaluating performance on 1000 random points...")
print ("Random accuracy: 25%")
correct_count = 0

for i in range(1000):
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

print("Accuracy:", correct_count / 1000)
with open('test_log.json', "w") as f:
    json.dump(data, f, indent=2)
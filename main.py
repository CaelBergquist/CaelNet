# main.py

import numpy as np
from model import SimpleNN
from train import Trainer
from rule import is_inside
import json
from snapshot import draw_network
import os
from snapshot2 import draw_points


# -------------------------
# Step 0: Initialize parameters
# --------------------
# -----
# Set hyperparameters
learning_rate = 0.001
num_iterations = 300000#0000
log_step = 10000
log = False
log2 = True
#batch_size = 32  # Optional: can do one sample at a time to start

# Create the model
model = SimpleNN(input_dim=2, hidden_dim=4, output_dim=1)
WEIGHTS_JSON = "weights.json"
TEST_LOG = "test_log.json"
NUM_TEST = 10000

if not os.path.exists(WEIGHTS_JSON):
    raise FileNotFoundError(f"{WEIGHTS_JSON} not found. Create it with your reset script first.")

with open(WEIGHTS_JSON, "r") as f:
    w = json.load(f)



# Create the trainer
trainer = Trainer(model, learning_rate)

# Load weights from JSON
with open("weights.json", "r") as f:
    weights = json.load(f)

if log2:
    with open(TEST_LOG, "w") as f:
        json.dump([], f)

data = []

# Convert lists back to numpy arrays and assign
model.W1 = np.array(weights["W1"])
model.b1 = np.array(weights["b1"])
model.W2 = np.array(weights["W2"])
model.b2 = np.array(weights["b2"])




# -------------------------
# Step 2: Training loop
# -------------------------


if __name__ == "__main__":

    print("Starting training...")
    print("Training for ", num_iterations, "iterations")
    total_correct = 0
    for iteration in range(num_iterations):
        
        if log and iteration % (int)(num_iterations/log_step) == 0:
            draw_network(step=iteration, accuracy=total_correct/((int)(num_iterations/log_step)) if iteration>0 else 0, 
                         weights_input_hidden=model.W1, bias_hidden=model.b1, weights_hidden_output=model.W2, bias_output=model.b2)
            total_correct = 0

        # x = random point from [0,0] to [1,1]
        x = np.array(np.random.rand(2))  
        
        y = np.array(is_inside(x[0], x[1]))

        loss, correct = trainer.forward_and_backward(x, y)
        if correct:
            total_correct += 1
            
        if log2:
            if correct:
                entry = {"x": float(x[0]), "y": float(x[1]), "correct": "y"}
                data.append(entry)
            else:
                entry = {"x": float(x[0]), "y": float(x[1]), "correct": "n"}
                data.append(entry)
            
            if iteration % log_step == 0 or iteration == num_iterations - 1:
                with open(TEST_LOG, "w") as f:
                    json.dump(data, f, indent=2)
                draw_points(step=iteration)
                data = []
            
        



    weights_to_save = {
        "W1": model.W1.tolist(),
        "b1": model.b1.tolist(),
        "W2": model.W2.tolist(),
        "b2": model.b2.tolist()
    }
    with open("weights.json", "w") as f:
        json.dump(weights_to_save, f)

    # -------------------------
    # Step 4: Test / Evaluate model
    # -------------------------
    # 1. Generate a grid of points (optional) to visualize predictions
    # 2. Run model.forward(x) for each point
    # 3. Compare prediction vs rule
    # 4. Optionally print accuracy or visualize heatmap

    print("Training complete. Weights saved to weights.json")
    print ("evaluating performance on 1000 random points...")
    print ("Random accuracy: 25%")
    correct_count = 0

    for i in range(1000):
        x = np.array([np.random.rand(), np.random.rand()])  
        y_true = is_inside(x[0], x[1])
        y_pred = model.forward(x)
        y_pred_label = 1 if y_pred >= 0.5 else 0
        if y_true == y_pred_label:
            correct_count += 1

    print("Model Accuracy:", correct_count / 1000)
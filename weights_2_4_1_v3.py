import numpy as np
import json

# He initialization for ReLU to prevent early training bias
def he_init(shape):
    fan_in = shape[1]  # number of input connections
    return (np.random.randn(*shape) * np.sqrt(2 / fan_in))*20

def logit(p):
    return np.log(p / (1 - p))

p = 0.25

weights = {
    "W1": he_init((4, 2)).tolist(),  # hidden × input
    "b1": np.zeros((4, 1)).tolist(), # hidden biases
    "W2": he_init((1, 4)).tolist(),  # output × hidden
    "b2": np.zeros((1, 1)).tolist(),  # output bias


    #"b2": np.zeros((1, 1)).tolist()  # output bias
}


# Save to JSON
with open("weights.json", "w") as f:

    json.dump(weights, f)

print("Random weights JSON file created!")

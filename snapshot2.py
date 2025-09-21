import json
import os
import matplotlib.pyplot as plt

def draw_points(step=0):
    correct = 0
    total = 0
    # Load test log
    with open("test_log.json", "r") as f:
        data = json.load(f)

    # Separate X, Y, and correctness
    Xs, Ys, colors = [], [], []
    for row in data:
        Xs.append(row["x"])
        Ys.append(row["y"])
        total += 1
        if row["correct"].lower() == "y":
            colors.append("green")
            correct += 1
        else:
            colors.append("red")

    # Create figure
    plt.figure(figsize=(6,6))
    
    # Draw 1x1 outer square
    plt.plot([0,1,1,0,0], [0,0,1,1,0], "k-")
    
    # Draw 0.5x0.5 inner square centered at (0.5,0.5)
    plt.plot([0.25,0.75,0.75,0.25,0.25], [0.25,0.25,0.75,0.75,0.25], "k--")
    
    # Plot points
    plt.scatter(Xs, Ys, c=colors, s=10, alpha=0.6)

    # Formatting
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(f"Model Evaluation (Step {step}) - Accuracy: {correct}/{total} = {correct/total:.2%}i ")

    # Save image
    os.makedirs("images2", exist_ok=True)
    plt.savefig(f"images2/snapshot_{(int(step/100)):05d}.png")
    plt.close()

if __name__ == "__main__":
    draw_points(step=0)

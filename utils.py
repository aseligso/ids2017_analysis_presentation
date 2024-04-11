# utils.py
import os
import matplotlib.pyplot as plt

def save_metrics(name, accuracy, precision, recall, directory="results"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, f"{name}_metrics.txt")
    with open(filepath, "w") as f:
        f.write(f"Model: {name}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")

def save_decision_tree(dot_data, filename="decision_tree.dot", directory="results"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    with open(filepath, "w") as f:
        f.write(dot_data)

def save_visualization(fig, filename, directory="results"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    fig.savefig(filepath)

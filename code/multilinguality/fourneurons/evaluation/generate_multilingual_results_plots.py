import json
import os
import matplotlib.pyplot as plt

RESULTS_PATH = "/scratch/results/multilingual/sft_epochs.json"
X_KEY = ["epoch", "step"]
Y_KEYS = ["mmmlu_pass@1", "mmmlu_pass@8", "boxed_format_compliance_pct", "xcopa_pass@1", "mmmlu_prox_pass@1"]

def plot_results_from_json(json_path, x_key, y_keys, title, save_path):
    # Plot x_key vs each y_key (access each element of the json)
    with open(json_path) as f:
        data = json.load(f)
    plt.figure(figsize=(10, 6))
    for y_key in y_keys:
        x_values = [entry[x_key[0]] for entry in data]
        y_values = [entry[y_key] for entry in data]
        if y_key == "boxed_format_compliance_pct":
            y_values = [y / 100 for y in y_values]  # Convert to percentage
        plt.plot(x_values, y_values, label=y_key)
    plt.xlabel(x_key[0])
    plt.ylabel("Value")
    plt.title(title)
    plt.ylim(0, 1)  # Set y-axis to [0, 1] for better comparison
    plt.legend()
    plt.grid()
    plt.savefig(save_path)

if __name__ == "__main__":
    plot_results_from_json(RESULTS_PATH, X_KEY, Y_KEYS, title="Multilingual SFT Training", save_path="/scratch/results/multilingual/sft_epochs.png")
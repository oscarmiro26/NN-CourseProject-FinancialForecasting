import json
import matplotlib.pyplot as plt

def plot_losses(log_file_path):
    with open(log_file_path, 'r') as log_file:
        results = json.load(log_file)
    
    for result in results:
        lookback = result['lookback']
        lr = result['learning_rate']
        num_layers = result['num_layers']
        num_nodes = result['num_nodes']
        train_losses = result['train_losses']
        val_losses = result['val_losses']

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f"Losses for Lookback={lookback}, LR={lr}, Layers={num_layers}, Nodes={num_nodes}")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    model = input().upper()
    file_path = f'../{model}/grid_search_results.json'
    plot_losses(file_path)

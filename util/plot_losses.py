import json
import matplotlib.pyplot as plt

def plot_top_model_losses(log_file_path):
    with open(log_file_path, 'r') as log_file:
        results = json.load(log_file)
    
    # Organize results by lookback value
    lookback_results = {}
    for result in results:
        lookback = result['lookback']
        val_losses = result['val_losses']
        min_val_loss = min(val_losses)
        if lookback not in lookback_results:
            lookback_results[lookback] = {'val_losses': val_losses, 'min_val_loss': min_val_loss}
        else:
            if min_val_loss < lookback_results[lookback]['min_val_loss']:
                lookback_results[lookback] = {'val_losses': val_losses, 'min_val_loss': min_val_loss}

    plt.figure(figsize=(10, 6))
    
    for lookback, data in lookback_results.items():
        plt.plot(data['val_losses'], label=f'Lookback={lookback}')

    plt.title("Validation Losses for Top Models of Each Lookback")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    model = input().upper()
    file_path = f'../{model}/full_grid_search_results.json'
    plot_top_model_losses(file_path)

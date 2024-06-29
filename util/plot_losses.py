import json
import matplotlib.pyplot as plt

def read_losses(file_path):
    entries = []
    with open(file_path, 'r') as f:
        for line in f:
            entries.append(json.loads(line))
    return entries

def plot_losses(entries):
    for entry in entries:
        params = entry['params']
        train_losses = entry['train_losses']
        val_losses = entry['val_losses']

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f"Losses for params: {params}")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    model = input().upper()
    file_path = f'../{model}/losses.json'
    entries = read_losses(file_path)
    plot_losses(entries)

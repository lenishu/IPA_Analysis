import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_all_batches(file_path, output_dir_base='plots'):
    # Extract dataset name from the file path
    parts = file_path.replace('\\', '/').split('/')
    dataset = parts[1] if len(parts) > 1 else 'UnknownDataset'

    # Read data
    df = pd.read_csv(file_path, sep=r'\s+')

    # Unique batch sizes in the data
    batch_sizes = df['BS'].unique()

    plt.figure(figsize=(10, 6))

    # Plot each batch size
    for bs in sorted(batch_sizes):
        df_bs = df[df['BS'] == bs]
        plt.errorbar(df_bs['P%'], df_bs['IPA_Average'], yerr=df_bs['STD'],
                     fmt='o-', label=f'BS = {int(bs)}', capsize=5)

    # Title and labels
    title = f"{dataset}"
    plt.title(title)
    plt.xlabel('P%')
    plt.ylabel('IPA Average')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(output_dir_base, dataset)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{title}.png")
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    file_path = "data/L_1_SLP_FMNIST/prune_layers_ALL_output.txt"
    plot_all_batches(file_path)

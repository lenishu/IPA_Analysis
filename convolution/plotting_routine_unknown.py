import matplotlib.pyplot as plt
import pandas as pd
import os

base_dir = "C:/Users/Student/Desktop/physlab/convolution"
output_base_dir = "C:/Users/Student/Desktop/physlab/plots/dataset_compare"

PRUNE_LAYERS_OPTIONS = ['CONV', 'FHL', 'SHL', 'FHL+SHL', 'ALL']
ACCEPTABLE_PRUNE_PERCENTAGES = [i / 100 for i in range(0, 110, 10)]  # 0.0, 0.1, ... 1.0
ACCEPTABLE_BATCH_SIZES = [1024]

def infer_dataset(path):
    for ds in ['FMNIST', 'MNIST', 'CIFAR']:
        if ds in path.upper():
            return ds
    return 'Unknown'

def find_matching_files(base_dir, target_percentage, target_batch, target_layer):
    matching_files = {}
    target_percentage_str = str(target_percentage)
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith("run_0.txt") and \
               f"p-percentage_{target_percentage_str}" in root and \
               f"batch_size_{target_batch}" in root and \
               f"prune_layers_{target_layer}" in root:
                dataset = infer_dataset(root)
                if dataset != 'Unknown':
                    matching_files[dataset] = os.path.join(root, file)
    return matching_files

def load_ce_test_vs_bn(file_path):
    column_names = ["Current_Epoch", "Batch_Total", "CE_Train", "Accuracy(%)", "CE_TEST", "Batch_Number"]
    df = pd.read_csv(file_path, sep=r'\s+', skiprows=1, names=column_names, engine='python')

    df = df[df["CE_Train"] != "--"]
    df["CE_TEST"] = pd.to_numeric(df["CE_TEST"], errors='coerce')
    df["Batch_Number"] = pd.to_numeric(df["Batch_Number"], errors='coerce')
    df = df.dropna(subset=["CE_TEST", "Batch_Number"])

    return df["Batch_Number"], df["CE_TEST"]

def plot_by_pruning_percentage():
    for batch_size in ACCEPTABLE_BATCH_SIZES:
        for layer in PRUNE_LAYERS_OPTIONS:
            out_dir = os.path.join(output_base_dir, str(batch_size), layer)
            os.makedirs(out_dir, exist_ok=True)

            for idx, p in enumerate(ACCEPTABLE_PRUNE_PERCENTAGES):
                matching_files = find_matching_files(
                    base_dir, target_percentage=str(p), target_batch=str(batch_size), target_layer=layer)

                if not matching_files:
                    print(f"No files for batch={batch_size}, layer={layer}, pruning%={p}")
                    continue

                plt.figure(figsize=(12, 8))
                found_data = False

                for dataset in ['MNIST', 'FMNIST', 'CIFAR']:
                    path = matching_files.get(dataset)
                    if path:
                        try:
                            bn, ce_test = load_ce_test_vs_bn(path)
                            plt.plot(bn, ce_test, label=dataset)
                            found_data = True
                        except Exception as e:
                            print(f"Error loading {path}: {e}")

                if not found_data:
                    print(f"No valid data to plot for batch={batch_size}, layer={layer}, pruning%={p}")
                    plt.close()
                    continue

                plt.title(f"CE_TEST vs Batch_Number\nBatch Size: {batch_size}, Prune Layer: {layer}, Prune %: {p}")
                plt.xlabel("Batch Number")
                plt.ylabel("CE_TEST")
                plt.legend(loc='best', fontsize='medium')
                plt.grid(True)
                plt.tight_layout()

                filename = f"p-{idx}.png"
                out_path = os.path.join(out_dir, filename)
                plt.savefig(out_path)
                plt.close()
                print(f"Saved plot: {out_path}")

plot_by_pruning_percentage()

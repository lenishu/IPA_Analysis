import os
import math
import pandas as pd
import matplotlib.pyplot as plt

def collect_data(root_folder='data'):
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for fname in filenames:
            if fname.endswith('_output.txt') and fname.startswith('prune_layers_'):
                file_paths.append(os.path.join(dirpath, fname))
    return file_paths

def parse_metadata(file_path):
    parts = file_path.replace('\\', '/').split('/')
    dataset = None
    for part in parts:
        if part.startswith('Conv-'):
            dataset = part.split('-')[1].split('_')[0]
    layer_type = None
    for part in parts:
        if part.startswith('prune_layers_'):
            layer_type = part.replace('prune_layers_', '')
    return dataset, layer_type

def find_global_ylim(file_paths):
    global_min, global_max = float('inf'), float('-inf')
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path, sep=r'\s+')
            lower = (df['IPA_Average'] - df['STD']).min()
            upper = (df['IPA_Average'] + df['STD']).max()
            global_min = min(global_min, lower)
            global_max = max(global_max, upper)
        except Exception:
            continue
    return global_min, global_max

def plot_all_as_subplots(root_folder='data', output_dir='plots_IPA_VS_BN_combined'):
    file_paths = collect_data(root_folder)
    if not file_paths:
        print("No valid files found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    num_files = len(file_paths)
    cols = 3
    rows = math.ceil(num_files / cols)
    ymin, ymax = find_global_ylim(file_paths)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), squeeze=False)
    for idx, file_path in enumerate(file_paths):
        row, col = divmod(idx, cols)
        ax = axes[row][col]
        dataset, layer_type = parse_metadata(file_path)
        if dataset is None or layer_type is None:
            ax.set_visible(False)
            continue
        try:
            df = pd.read_csv(file_path, sep=r'\s+')
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            ax.set_visible(False)
            continue
        df_64 = df[df['BS'] == 64]
        df_1024 = df[df['BS'] == 1024]

        if not df_64.empty:
            ax.errorbar(df_64['P%'], df_64['IPA_Average'], yerr=df_64['STD'],
                        fmt='o-', label='BS = 64', capsize=5)
        if not df_1024.empty:
            ax.errorbar(df_1024['P%'], df_1024['IPA_Average'], yerr=df_1024['STD'],
                        fmt='s-', label='BS = 1024', capsize=5)

        ax.set_title(f"Conv-{dataset}-{layer_type}")
        ax.set_xlabel('P%')
        ax.set_ylabel('IPA Average')
        # ax.set_ylim(ymin, ymax)
        ax.grid(True)
        ax.legend()

    # Hide unused axes
    for idx in range(num_files, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'combined_subplot_1.png')
    plt.savefig(save_path)
    print(f"Saved all subplots to: {save_path}")
    plt.close()

if __name__ == "__main__":
    plot_all_as_subplots()

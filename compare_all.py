import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools

def extract_metadata(file_path):
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

def plot_all_prune_outputs(root_folder='data', output_image='combined_prune_plot.png'):
    plt.figure(figsize=(12, 8))
    color_map = cm.get_cmap('tab20')  # Up to 20 unique colors
    markers = ['o', 's', 'v', 'D', '^', '>', '<', '*', 'p', 'h']
    style_iter = itertools.cycle(itertools.product(markers, range(color_map.N)))

    plotted = 0

    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            if fname.endswith('_output.txt') and fname.startswith('prune_layers_'):
                file_path = os.path.join(dirpath, fname)
                dataset, layer_type = extract_metadata(file_path)

                if dataset is None or layer_type is None:
                    print(f"Skipping {file_path} - missing dataset/layer_type")
                    continue

                try:
                    df = pd.read_csv(file_path, sep=r'\s+')
                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")
                    continue

                for bs in [64, 1024]:
                    df_bs = df[df['BS'] == bs]
                    if df_bs.empty:
                        continue

                    marker, color_idx = next(style_iter)
                    label = f"{dataset}-{layer_type} (BS={bs})"

                    plt.errorbar(df_bs['P%'], df_bs['IPA_Average'], yerr=df_bs['STD'],
                                 fmt=marker + '-', label=label, capsize=4, color=color_map(color_idx))

                    plotted += 1

    if plotted == 0:
        print("No valid data to plot.")
        return

    plt.title("Combined IPA vs Pruning Percentage")
    plt.xlabel("Pruning Percentage (P%)")
    plt.ylabel("IPA Average")
    plt.grid(True)
    plt.legend(loc='best', fontsize='small', ncol=2)
    plt.tight_layout()
    plt.savefig(output_image)
    print(f"Combined plot saved to: {output_image}")
    plt.close()

if __name__ == "__main__":
    plot_all_prune_outputs()

import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_prune_output(file_path, output_dir_base='plots'):
    parts = file_path.replace('\\', '/').split('/')

    # Dataset: folder starting with 'Conv-' (e.g. Conv-FMIST_IPA_output_1)
    dataset = None
    for part in parts:
        if part.startswith('Conv-'):
            # Grab after 'Conv-' up to first underscore or end
            dataset = part.split('-')[1].split('_')[0]

    # Layer type: folder starting with prune_layers_
    layer_type = None
    for part in parts:
        if part.startswith('prune_layers_'):
            layer_type = part.replace('prune_layers_', '')

    if dataset is None or layer_type is None:
        print(f"Skipping {file_path} - could not parse dataset/layer type")
        return

    try:
        df = pd.read_csv(file_path, sep=r'\s+')
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return

    df_64 = df[df['BS'] == 64]
    df_1024 = df[df['BS'] == 1024]

    if df_64.empty and df_1024.empty:
        print(f"No batch size 64 or 1024 data in {file_path}")
        return

    output_dir = os.path.join(output_dir_base, f"{dataset}_{layer_type}")
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    if not df_64.empty:
        plt.errorbar(df_64['P%'], df_64['IPA_Average'], yerr=df_64['STD'],
                     fmt='o-', label='BS = 64', capsize=5)
    if not df_1024.empty:
        plt.errorbar(df_1024['P%'], df_1024['IPA_Average'], yerr=df_1024['STD'],
                     fmt='s-', label='BS = 1024', capsize=5)

    title = f"Random-Pruning-Conv-{dataset}-{layer_type}"
    plt.title(title)
    plt.xlabel('P%')
    plt.ylabel('IPA Average')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = f"{title}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.close()

def main(root_folder='data'):
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for fname in filenames:
            if fname.endswith('_output.txt') and fname.startswith('prune_layers_'):
                file_path = os.path.join(dirpath, fname)
                plot_prune_output(file_path)

if __name__ == "__main__":
    main()

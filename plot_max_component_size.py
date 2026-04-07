import ast
import os
import re
import sys
import matplotlib.pyplot as plt

def load_data(file_path):
    with open(file_path, 'r') as f:
        raw = f.read()
    # Strip leading line numbers if present (format: "1\t{...},\n2\t{...},")
    lines = []
    for line in raw.splitlines():
        parts = line.split('\t', 1)
        lines.append(parts[-1] if len(parts) == 2 else line)
    data = ast.literal_eval('\n'.join(lines))
    return data

def title_from_filename(file_path):
    name = os.path.splitext(os.path.basename(file_path))[0]
    # e.g. "[625,25]_HGP_code_peeling_cluster_decoder_performance_data_5000_trials_v5"
    # Extract code label and trial count using the known naming convention
    m = re.match(r'(\[[\d,]+\]_HGP_code).*?(\d+)_trials', name)
    if m:
        code_label = m.group(1).replace('_', ' ')
        trials = m.group(2)
        return f"{code_label} — Cluster-only decoder\nMax component size vs erasure rate  ({trials} trials)"
    # Fallback: use the raw filename
    return name.replace('_', ' ')


def plot(data, output_file=None, data_file=None):
    # Filter out entries where average is 0 and std is 0 (all trials failed, no data)
    valid = [d for d in data if d['average_cluster_only_max_component_size'] > 0]

    x   = [d['erasure_rate']                          for d in valid]
    y   = [d['average_cluster_only_max_component_size'] for d in valid]
    err = [d['std_cluster_only_max_component_size']    for d in valid]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(x, y, yerr=err,
                fmt='o-', capsize=2, capthick=.5,
                linewidth=.5, markersize=2,
                label='Mean ± Std dev (Largest component size without peeling)')

    y_max = [d['max_cluster_only_max_component_size'] for d in valid]
    ax.plot(x, y_max, 's--', color='red', linewidth=.5, markersize=2,
            label='Max (Largest component size without peeling)')

    valid_mc = [d for d in data if d['average_max_component_size'] > 0]
    if valid_mc:
        x_mc   = [d['erasure_rate']              for d in valid_mc]
        y_mc   = [d['average_max_component_size'] for d in valid_mc]
        err_mc = [d['std_max_component_size']     for d in valid_mc]
        ax.errorbar(x_mc, y_mc, yerr=err_mc,
                    fmt='o-', color='green', capsize=2, capthick=.5,
                    linewidth=.5, markersize=2,
                    label='Mean ± Std dev (Largest component size after peeling)')

        y_max_mc = [d['max_max_component_size'] for d in valid_mc]
        ax.plot(x_mc, y_max_mc, 's--', color='orange', linewidth=.5, markersize=2,
                label='Max (Largest component size after peeling)')

    ax.set_xlabel('Erasure rate', fontsize=13)
    ax.set_ylabel('Max component size', fontsize=13)
    title = title_from_filename(data_file) if data_file else 'Cluster-only decoder\nMax component size vs erasure rate'
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()

    if output_file:
        base = os.path.splitext(output_file)[0]
        fig.savefig(base + '.png', dpi=300)
        print(f"Saved to {base}.png")
        fig.savefig(base + '.pdf')
        print(f"Saved to {base}.pdf")
    else:
        plt.show()

if __name__ == '__main__':
    data_file = (sys.argv[1] if len(sys.argv) > 1 else
        "Peeling_Cluster_Decoder_Array_Job_Folder_[625,25]_HGP_code/"
        "[625,25]_HGP_code_peeling_cluster_decoder_performance_data_5000_trials_v5.txt")
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    data = load_data(data_file)
    plot(data, output_file, data_file)

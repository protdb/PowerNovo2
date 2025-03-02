from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline


def sort_order(subset):
    if subset == 'Non_Trypic':
        return 0
    elif subset.startswith('MNIST'):
        return 1
    else:
        return 2


def plot_metrics(input_file, key='PEPTIDE PRECISION'):


    df = pd.read_csv(input_file)
    mean_values = df.groupby(['SERVICE', 'SUBSET'])[key].mean().reset_index()
    mean_values['sort_order'] = mean_values['SUBSET'].apply(sort_order)
    mean_values = mean_values.sort_values(by=['SERVICE', 'sort_order', 'SUBSET'])
    mean_values.drop(columns=['sort_order'], inplace=True)

    unique_subsets = mean_values['SUBSET'].unique()


    colors = ['olive', 'indigo', 'royalblue', 'teal', 'r', 'navy']
    markers = ['p', 'o', 'P', 'D', 's', 'P', '<']


    plt.figure(figsize=(15, 8.5), dpi=100)

    for idx, service in enumerate(mean_values['SERVICE'].unique()):
        service_data = mean_values[mean_values['SERVICE'] == service]
        x = np.arange(len(service_data['SUBSET']))  # Индекс для оси X


        x_smooth = np.linspace(x.min(), x.max(), 500)
        spl = make_interp_spline(x, service_data[key], k=3)
        y_smooth = spl(x_smooth)


        plt.plot(x_smooth, y_smooth, color=colors[idx % len(colors)], label=None, alpha=0.5)
        plt.scatter(x, service_data[key], color=colors[idx % len(colors)],
                    marker=markers[idx % len(markers)], s=100, edgecolor='black', label=service)

        y = service_data[key].values
        for i, value in enumerate(y):
            plt.annotate(f"{value:.2f}", (x[i], y[i]), textcoords="offset points", xytext=(5,  1 * idx), ha='left')


    axis_title = []

    for s in unique_subsets:
        if s == 'Non_Trypic':
            axis_title.append("Non-Tryptic")

        elif 'MNIST' in s:
            s = s.replace('MNIST_', '')
            axis_title.append(s)
        else:
            axis_title.append(s.strip())


    plt.xticks(np.arange(len(axis_title)), axis_title, rotation=60, fontsize=14)
    plt.ylabel(key, fontsize=14)
    plt.legend(title='DE NOVO SERVICE', frameon=False, fontsize=14)
    plt.grid(alpha=0.2, axis='x', linestyle='--')
    plt.tight_layout()

    for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)



    output_file = Path(input_file).parent /  f'{Path(input_file).stem}_{key}.png'
    plt.savefig(output_file)


file = "/Data/benchmark/results/denovo_results_(len=30).csv"

if __name__ == '__main__':
    plot_metrics(file)
    plot_metrics(file, 'AA PRECISION')

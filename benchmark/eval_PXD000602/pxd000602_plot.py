import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file_path = "/Data/benchmark/results_pxd/coverage_total_conc.csv"
data = pd.read_csv(file_path)


high_concentration = data[data["Concentration"].isin([50000, 5000])]
medium_concentration = data[data["Concentration"].isin([500, 50])]
low_concentration = data[data["Concentration"].isin([5, 0.5])]

# Все сервисы (колонки данных)
services = ["CASANOVO",  "PEAKS", "PEPNET", "POWERNOVO1", "POWERNOVO2", "MAXQUANT",]
colors = ['olive', 'indigo', 'royalblue', 'teal', 'r', 'black']
colors = {s: colors[i] for i, s in enumerate(services)}


plot1 = [(8, 8), (5, 8), (8, 8), (8, 8), (8, 8),  (6, 8)]
plot2 = [(8, 8), (2, 8), (7, 8), (5, 8), (8, 8), (6, 8)]
plot3 = [(6, 8), (1, 8), (7, 8), (1, 8), (5, 8), (6, 8)]
plot4 = [(7, 8), (3, 8), (5, 8), (1, 8), (5, 8), (1, 8)]
plot5 = [(7, 8), (4, 8), (7, 8), (1, 8), (7, 8), (0, 8)]
plot6 = [(7, 8), (4, 8), (6, 8), (0, 8), (7, 8), (0, 8)]


def plot_bar_chart(data_subset, title, ax, sb):
    # Перебираем белки и строим бар-чарты
    proteins = data_subset["PROTEIN_ID"].unique()
    x = range(len(proteins))

    for i, service in enumerate(services):

        heights = data_subset.groupby("PROTEIN_ID")[service].max()

        ax.bar([p + i * 0.15 for p in x],   heights, width=0.15, label=f'{service} ({sb[i][0]}/{sb[i][1]})',
               color=colors[service])

        tx = heights.to_numpy()
        tx = np.round(tx, 1)

        for p in x:
            value = tx[p]
            if value == 0:
                continue
            ax.text(p + i * 0.15 - 0.09, tx[p] + 0.5, f'{value:.1f}', fontsize=10)





    ax.set_title(title, fontsize=16)
    ax.set_xticks([p + 0.35 for p in x])  # Центровка меток X
    ax.set_xticklabels(proteins, rotation=45, ha="right", fontsize=12)
    ax.set_ylabel("Protein coverage %", fontsize=12)
    ax.legend(fontsize=12, loc='upper left', frameon=False)

    for pos in ['right', 'top']:
        ax.spines[pos].set_visible(False)

fig, axes = plt.subplots(3, 2, figsize=(20, 15), sharey=True)


plot_bar_chart(high_concentration[high_concentration["Concentration"] == 50000],
               "High Concentration (50,000 fmol)", axes[0, 0], sb=plot1)
plot_bar_chart(high_concentration[high_concentration["Concentration"] == 5000],
               "High Concentration (5,000 fmol)", axes[0, 1], sb=plot2)

plot_bar_chart(medium_concentration[medium_concentration["Concentration"] == 500],
               "Medium Concentration (500 fmol)", axes[1, 0], sb=plot3)

plot_bar_chart(medium_concentration[medium_concentration["Concentration"] == 50],
               "Medium Concentration (50 fmol)", axes[1, 1], sb=plot4)


plot_bar_chart(low_concentration[low_concentration["Concentration"] == 5],
               "Low Concentration (5 fmol)", axes[2, 0], sb=plot5)
plot_bar_chart(low_concentration[low_concentration["Concentration"] == 0.5],
               "Low Concentration (0.5 fmol)", axes[2, 1], sb=plot6)

plt.tight_layout()
plt.savefig(file_path.replace('csv', 'png'))

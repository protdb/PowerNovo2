import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


def aa_precision_recall(
    aa_scores_correct,
    aa_scores_all,
    n_aa_total: int,
    threshold: float,
):
    """
    Calculate amino acid level precision and recall at a given score threshold.

    Parameters
    ----------
    aa_scores_correct : List[float]
        Amino acids scores for the correct amino acids predictions.
    aa_scores_all : List[float]
        Amino acid scores for all amino acids predictions.
    n_aa_total : int
        The total number of amino acids in the predicted peptide sequences.
    threshold : float
        The amino acid score threshold.

    Returns
    -------
    aa_precision: float
        The number of correct amino acid predictions divided by the number of
        predicted amino acids.
    aa_recall: float
        The number of correct amino acid predictions divided by the total number
        of amino acids.
    """
    n_aa_correct = sum([score > threshold for score in aa_scores_correct])
    n_aa_predicted = sum([score > threshold for score in aa_scores_all])
    return n_aa_correct / n_aa_predicted, n_aa_correct / n_aa_total


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


def plot_per_organism(data, output_file):
    """
    Построить Precision-Recall кривые для каждого организма на отдельных субплотах.

    :param data: словарь формата {service: {subset_name: {'aa_scores_correct': [...], 'aa_scores_all': [...]}}}
                 where:
                 - service: name of de novo service (e.g. "Service1")
                 - subset_name: name of organism (e.g. "Organism1")
                 - aa_scores_correct: list of scores for correctly predicted amino acids
                 - aa_scores_all: list of scores for all amino acids
    """
    # Получение полного списка подмножеств (organisms/subsets)
    subsets = set()
    for service_data in data.values():
        subsets.update(service_data.keys())
    subsets = sorted(subsets)  # Упорядочим для единообразия

    # Количество subplots определяется количеством организмов
    num_subplots = len(subsets)
    rows = int(np.ceil(np.sqrt(num_subplots)))  # для сетки subplots (по возможности квадратной)
    cols = int(np.ceil(num_subplots / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 10), squeeze=False)
    axes = axes.flatten()  # Преобразуем в 1D массив для удобного доступа

    # Перебираем каждое подмножество (организм)
    for idx, subset_name in enumerate(subsets):
        ax = axes[idx]
        for service, service_data in data.items():
            if subset_name in service_data:
                scores = service_data[subset_name]
                aa_scores_correct = np.array(scores['aa_scores_correct'])
                aa_scores_all = np.array(scores['aa_scores_all'])

                # Создаем Precision-Recall кривую
                y_true = np.concatenate([np.ones_like(aa_scores_correct), np.zeros_like(aa_scores_all)])
                y_scores = np.concatenate([aa_scores_correct, aa_scores_all])
                precision, recall, _ = precision_recall_curve(y_true, y_scores)

                # Рисуем кривую
                ax.plot(recall, precision, label=f'{service}')

        # Настройки для текущего subplot
        ax.set_title(f"Organism: {subset_name}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()

    # Для неиспользованных подграфиков скрываем оси
    for i in range(len(subsets), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)

input_file = '/Data/benchmark/results/auc_data.pkl'
if __name__ == '__main__':
    with open(input_file, 'rb') as fh:
        data = pickle.load(fh)

    output_file = Path(input_file).parent / 'pr_curves.png'

    plot_per_organism(data, output_file)

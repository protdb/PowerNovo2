"""Methods to evaluate peptide-spectrum predictions."""
import os
import pickle
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from powernovo2.config.default_config import aa_residues
from spectrum_utils.utils import mass_diff


def aa_match_prefix(
        peptide1: List[str],
        peptide2: List[str],
        aa_dict: Dict[str, float],
        cum_mass_threshold: float = 0.5,
        ind_mass_threshold: float = 0.1,
) -> Tuple[np.ndarray, bool]:
    """
    Find the matching prefix amino acids between two peptide sequences.
    """
    aa_matches = np.zeros(max(len(peptide1), len(peptide2)), np.bool_)
    i1, i2, cum_mass1, cum_mass2 = 0, 0, 0.0, 0.0
    while i1 < len(peptide1) and i2 < len(peptide2):
        aa_mass1 = aa_dict.get(peptide1[i1], 0)
        aa_mass2 = aa_dict.get(peptide2[i2], 0)
        if (
                abs(mass_diff(cum_mass1 + aa_mass1, cum_mass2 + aa_mass2, True))
                < cum_mass_threshold
        ):
            aa_matches[max(i1, i2)] = (
                    abs(mass_diff(aa_mass1, aa_mass2, True)) < ind_mass_threshold
            )
            i1, i2 = i1 + 1, i2 + 1
            cum_mass1, cum_mass2 = cum_mass1 + aa_mass1, cum_mass2 + aa_mass2
        elif cum_mass2 + aa_mass2 > cum_mass1 + aa_mass1:
            i1, cum_mass1 = i1 + 1, cum_mass1 + aa_mass1
        else:
            i2, cum_mass2 = i2 + 1, cum_mass2 + aa_mass2
    return aa_matches, aa_matches.all()


def aa_match_prefix_suffix(
        peptide1: List[str],
        peptide2: List[str],
        aa_dict: Dict[str, float],
        cum_mass_threshold: float = 0.5,
        ind_mass_threshold: float = 0.1,
) -> Tuple[np.ndarray, bool]:
    """
    Find the matching prefix and suffix amino acids between two peptide sequences.
    """
    aa_matches, pep_match = aa_match_prefix(
        peptide1, peptide2, aa_dict, cum_mass_threshold, ind_mass_threshold
    )
    if pep_match:
        return aa_matches, pep_match

    i1, i2 = len(peptide1) - 1, len(peptide2) - 1
    i_stop = np.argwhere(~aa_matches)[0]
    cum_mass1, cum_mass2 = 0.0, 0.0
    while i1 >= i_stop and i2 >= i_stop:
        aa_mass1 = aa_dict.get(peptide1[i1], 0)
        aa_mass2 = aa_dict.get(peptide2[i2], 0)
        if (
                abs(mass_diff(cum_mass1 + aa_mass1, cum_mass2 + aa_mass2, True))
                < cum_mass_threshold
        ):
            aa_matches[max(i1, i2)] = (
                    abs(mass_diff(aa_mass1, aa_mass2, True)) < ind_mass_threshold
            )
            i1, i2 = i1 - 1, i2 - 1
            cum_mass1, cum_mass2 = cum_mass1 + aa_mass1, cum_mass2 + aa_mass2
        elif cum_mass2 + aa_mass2 > cum_mass1 + aa_mass1:
            i1, cum_mass1 = i1 - 1, cum_mass1 + aa_mass1
        else:
            i2, cum_mass2 = i2 - 1, cum_mass2 + aa_mass2
    return aa_matches, aa_matches.all()


def aa_match(
        peptide1: List[str],
        peptide2: List[str],
        aa_dict: Dict[str, float],
        cum_mass_threshold: float = 0.5,
        ind_mass_threshold: float = 0.1,
        mode: str = "best",
) -> Tuple[np.ndarray, bool]:
    """
    Find the matching amino acids between two peptide sequences.
    """
    if mode == "best":
        return aa_match_prefix_suffix(
            peptide1, peptide2, aa_dict, cum_mass_threshold, ind_mass_threshold
        )
    elif mode == "forward":
        return aa_match_prefix(
            peptide1, peptide2, aa_dict, cum_mass_threshold, ind_mass_threshold
        )
    elif mode == "backward":
        aa_matches, pep_match = aa_match_prefix(
            list(reversed(peptide1)),
            list(reversed(peptide2)),
            aa_dict,
            cum_mass_threshold,
            ind_mass_threshold,
        )
        return aa_matches[::-1], pep_match
    else:
        raise ValueError("Unknown evaluation mode")


def aa_match_batch(
        peptides1: Iterable,
        peptides2: Iterable,
        aa_dict: Dict[str, float],
        cum_mass_threshold: float = 0.5,
        ind_mass_threshold: float = 0.1,
        mode: str = "best",
) -> Tuple[List[Tuple[np.ndarray, bool]], int, int]:
    """
    Find the matching amino acids between multiple pairs of peptide sequences.
    """
    aa_matches_batch, n_aa1, n_aa2 = [], 0, 0
    for peptide1, peptide2 in zip(peptides1, peptides2):
        if isinstance(peptide1, str):
            peptide1 = re.split(r"(?<=.)(?=[A-Z])", peptide1)
        if isinstance(peptide2, str):
            peptide2 = re.split(r"(?<=.)(?=[A-Z])", peptide2)
        n_aa1, n_aa2 = n_aa1 + len(peptide1), n_aa2 + len(peptide2)
        aa_matches_batch.append(
            aa_match(
                peptide1,
                peptide2,
                aa_dict,
                cum_mass_threshold,
                ind_mass_threshold,
                mode,
            )
        )
    return aa_matches_batch, n_aa1, n_aa2


def aa_match_metrics(
        aa_matches_batch: List[Tuple[np.ndarray, bool]],
        n_aa_true: int,
        n_aa_pred: int,
) -> Tuple[float, float, float]:
    """
    Calculate amino acid and peptide-level evaluation metrics.
    """
    n_aa_correct = sum(
        [aa_matches[0].sum() for aa_matches in aa_matches_batch]
    )
    aa_precision = n_aa_correct / (n_aa_pred + 1e-8)
    aa_recall = n_aa_correct / (n_aa_true + 1e-8)
    pep_precision = sum([aa_matches[1] for aa_matches in aa_matches_batch]) / (
            len(aa_matches_batch) + 1e-8
    )
    return float(aa_precision), float(aa_recall), float(pep_precision)



def process_service_folder(service_path: Path, output_dir: Path, only_auc: bool = True):
    """
    Обрабатывает все файлы в папке одного сервиса и сохраняет результаты.
    """
    service_name = service_path.name
    print(f"Обработка сервиса: {service_name}")

    # Собираем все CSV файлы для этого сервиса
    csv_files = []
    for root, dirs, files in os.walk(service_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(Path(root) / file)

    if not csv_files:
        print(f"Не найдено CSV файлов для сервиса {service_name}")
        return

    # Данные для текущего сервиса
    service_data = {}
    results = {'SERVICE': [], 'FILE': [], 'SET': [], 'SUBSET': [],
               'AA PRECISION': [], 'AA RECALL': [], 'PEPTIDE PRECISION': []}
    true_peps = {'SERVICE': [], 'FILE': [], 'SET': [], 'SUBSET': [], 'TITLE': [], 'DENOVO': []}

    for file_path in tqdm(csv_files, desc=f"Обработка {service_name}"):
        # Извлекаем set_name и subset_name из пути
        parts = file_path.parts
        set_name = parts[-3] if len(parts) >= 3 else "unknown"
        subset_name = parts[-2] if len(parts) >= 2 else "unknown"

        try:
            df = pd.read_csv(file_path)
            true_peps_list = df['TITLE'].to_list()
            denovo_peps = df['DENOVO'].to_list()
            aa_scores = df['Positional Score'].to_list()

            result = aa_match_batch(true_peps_list, denovo_peps, aa_residues)

            if only_auc:
                # Обработка для AUC данных
                for i, res in enumerate(result[0]):
                    aa_matches, is_match = res
                    scores_f = aa_scores[i].split(' ')
                    scores_f = list(map(float, scores_f))

                    if subset_name not in service_data:
                        service_data[subset_name] = {
                            'aa_scores_correct': [],
                            'aa_scores_all': [],
                            'peptide_scores_correct': [],  # ДОБАВЛЕНО: скоры правильных пептидов
                            'peptide_scores_all': []  # ДОБАВЛЕНО: скоры всех пептидов
                        }

                    # Данные для аминокислот (остается как было)
                    if len(aa_matches) > 0:
                        for j in range(min(len(scores_f), len(aa_matches))):
                            score = scores_f[j]
                            score = np.nan_to_num(score)

                            if aa_matches[j]:
                                service_data[subset_name]['aa_scores_correct'].append(score)
                            service_data[subset_name]['aa_scores_all'].append(score)
                    else:
                        service_data[subset_name]['aa_scores_all'].append(0)

                    # ДОБАВЛЕНО: Данные для пептидов
                    if len(scores_f) > 0:
                        # Для пептида берем минимальный скор среди аминокислот
                        peptide_score = np.mean(scores_f) if len(scores_f) > 0 else 0
                        peptide_score = np.nan_to_num(peptide_score)

                        if is_match:  # Если пептид полностью правильный
                            service_data[subset_name]['peptide_scores_correct'].append(peptide_score)
                        service_data[subset_name]['peptide_scores_all'].append(peptide_score)

            else:
                # Обработка для обычных метрик (остается без изменений)
                for i, res in enumerate(result[0]):
                    aa_matches, is_match = res
                    if is_match:
                        true_peps['SERVICE'].append(service_name)
                        true_peps['FILE'].append(file_path.name)
                        true_peps['SET'].append(set_name)
                        true_peps['SUBSET'].append(subset_name)
                        true_peps['TITLE'].append(true_peps_list[i])
                        true_peps['DENOVO'].append(denovo_peps[i])

                aa_matches_batch, n_aa_true, n_aa_pred = result
                metrics = aa_match_metrics(aa_matches_batch, n_aa_true, n_aa_pred)
                aa_precision, aa_recall, pep_precision = metrics

                results['SERVICE'].append(service_name)
                results['FILE'].append(file_path.name)
                results['SET'].append(set_name)
                results['SUBSET'].append(subset_name)
                results['AA PRECISION'].append(aa_precision)
                results['AA RECALL'].append(aa_recall)
                results['PEPTIDE PRECISION'].append(pep_precision)

        except Exception as e:
            print(f"Ошибка при обработке файла {file_path}: {e}")
            continue

    # Сохраняем результаты для текущего сервиса
    if only_auc:
        # Сохраняем AUC данные в отдельный pkl файл
        auc_file = output_dir / f'auc_data_{service_name}.pkl'
        with open(auc_file, 'wb') as f:
            pickle.dump(service_data, f)
        print(f"Сохранен файл AUC данных: {auc_file}")
    else:
        # Сохраняем обычные метрики
        if results['SERVICE']:
            df_results = pd.DataFrame(results)
            results_file = output_dir / f'results_{service_name}.csv'
            df_results.to_csv(results_file, index=False)
            print(f"Сохранены результаты: {results_file}")

        if true_peps['SERVICE']:
            df_true_peps = pd.DataFrame(true_peps)
            true_peps_file = output_dir / f'true_peps_{service_name}.csv'
            df_true_peps.to_csv(true_peps_file, index=False)
            print(f"Сохранены true peptides: {true_peps_file}")

    # Очищаем память
    del service_data, results, true_peps
    return service_name




def calculate(dataset_root_folder: str, only_auc: bool = True):
    """    Основная функция обработки всех сервисов.    """
    dataset_path = Path(dataset_root_folder)
    output_dir = dataset_path.parent / 'processed_results_gt30'
    output_dir.mkdir(parents=True, exist_ok=True)

# Получаем список папок сервисов (первый уровень вложенности)
    service_folders = [f for f in dataset_path.iterdir() if f.is_dir()]

    print(f"Найдено сервисов для обработки: {len(service_folders)}")

    processed_services = []
    for service_folder in service_folders:
        try:
            service_name = process_service_folder(service_folder, output_dir, only_auc)
            processed_services.append(service_name)
            print(f"Завершена обработка сервиса: {service_name}")
        except Exception as e:
            print(f"Ошибка при обработке сервиса {service_folder.name}: {e}")
            continue

    print(f"Обработка завершена. Обработано сервисов: {len(processed_services)}")
    print("Список обработанных сервисов:", processed_services)


DATASET_PATH30 = "/home/dp/Data/benchmark/results/denovo_results_(len=30)"
DATASET_PATHgt_30 = "/home/dp/Data/benchmark/results/denovo_results_(len_gt_30)"

if __name__ == '__main__':
    calculate(DATASET_PATHgt_30, only_auc=False)

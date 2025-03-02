import math

import matplotlib.pyplot as plt
import pandas as pd


def plot_identity(input_file):

    data = pd.read_csv(input_file)

    colors = ['olive', 'indigo', 'royalblue', 'teal', 'r', 'navy']

    # 2. Определяем, какие столбцы являются метаинформацией (SUBSET и SERVICE), а какие порогами
    meta_columns = ['SUBSET', 'SERVICE']
    threshold_columns = [col for col in data.columns if col not in meta_columns]

    # Преобразуем числовые значения в столбцах порогов
    data[threshold_columns] = data[threshold_columns].apply(pd.to_numeric, errors='coerce')

    # 3. Вычисляем средние значения идентификаций для каждого порога
    # Группируем данные по метаинформации (SUBSET, SERVICE) и вычисляем средние значения по порогам
    grouped_data = data.groupby(meta_columns)[threshold_columns].mean().reset_index()

    # 4. Динамически определяем количество организмов (SUBSET) и количество рядов-колонок для сетки
    unique_subsets = grouped_data['SUBSET'].unique()  # уникальные организмы
    num_subsets = len(unique_subsets)  # количество организмов

    unique_subsets = ['Apis-mellifera', 'Bacillus-subtilis', 'Candidatus-endoloripes', 'H.-sapiens',
                      'Methanosarcina-mazei', 'Mus-musculus',
                      'Saccharomyces-cerevisiae', 'Solanum-lycopersicum', 'Vigna-mungo',
                      'MNIST_E.Coli', 'MNIST_H.Sapience', 'MNIST_M.Musculus', 'MNIST_Yeast','Non_Trypic']


    titles = ['nine-species:Apis-mellifera', 'nine-species:Bacillus-subtilis',
                      'nine-species:Candidatus-endoloripes', 'nine-species:H.-sapiens',
                      'nine-species:Methanosarcina-mazei', 'nine-species:Mus-musculus',
                      'nine-species:Saccharomyces-cerevisiae', 'nine-species:Solanum-lycopersicum', 'nine-species:Vigna-mungo',
                      'NIST:E.Coli', 'NIST:H.Sapience', 'NIST:M.Musculus', 'NIST:Yeast', 'MassIVE:Non-Tryptic']



    # Определим размеры сетки для субплотов
    num_cols = 3  # Можно зафиксировать количество колонок
    num_rows = math.ceil(num_subsets / num_cols)  # Количество строк в зависимости от данных

    # 5. Создаем фигуру и сетку субплотов
    fig, axes = plt.subplots(num_rows, num_cols,
                             figsize=(12, 4 * num_rows))  # Размер увеличивается пропорционально числу строк
    axes = axes.flatten()  # Преобразуем массив осей в одномерный для удобства

    # 6. Построение графиков
    for i, subset in enumerate(unique_subsets):
        ax = axes[i]  # Выбираем текущий subplot
        subset_data = grouped_data[grouped_data['SUBSET'] == subset]  # Фильтруем данные по организму

        # Для каждого сервиса строим кривую
        for j, service in enumerate(subset_data['SERVICE'].unique()):
            service_data = subset_data[subset_data['SERVICE'] == service]
            ax.plot(threshold_columns, service_data[threshold_columns].values.flatten(), label=service, marker='o',
                    color=colors[j])
            ax.set_xticks([str(i / 10) for i in range(0, 11)])

        # Настраиваем подписи и заголовки текущего графика
        ax.set_title(titles[i], fontsize=12)

        ax.set_ylabel('Proportion of peptides', fontsize=14)
        ax.set_xlabel('Peptide sequence identity', fontsize=14)
        ax.legend(title='DE NOVO SERVICE', frameon=False, fontsize=14)

    # Убираем лишние пустые субплоты, если организмов меньше, чем графиков в сетке
    for j in range(len(unique_subsets), len(axes)):
        fig.delaxes(axes[j])

    # 7. Настраиваем общий заголовок и отображаем графики
    plt.tight_layout()

    output_file = input_file.replace('csv', 'png')
    plt.savefig(output_file, dpi=300)

file = "/Data/benchmark/results/identity_(len=30).csv"

if __name__ == '__main__':
    plot_identity(file)


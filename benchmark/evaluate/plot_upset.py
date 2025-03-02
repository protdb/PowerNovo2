import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from upsetplot import UpSet, from_contents


def plot_upset(file_path):
    df = pd.read_csv(file_path)  # Если разделитель — табуляция (\t)
    matplotlib.rcParams.update({'font.size': 10})

    unique_peptides = df.groupby('SERVICE')['DENOVO'].apply(set)

    uq = len(set(df['DENOVO'].to_list()))
    print(uq)


    set_names = unique_peptides.index.to_list()
    set_names[-1], set_names[-2] = set_names[-2], set_names[-1]

    set_data = [unique_peptides[service] for service in set_names]


    data_dict = {name: peptides for name, peptides in zip(set_names, set_data)}
    upset_data = from_contents(data_dict)

    upset = UpSet(upset_data,
                  subset_size='count',
                  show_counts='%d',
                  min_subset_size=100,
                  facecolor='#21618C',
                  sort_categories_by=None)
    fig = plt.figure(figsize=(15, 6), dpi=300)
    upset.style_subsets(
        present='PowerNovo2',
        absent=['PowerNovo1', 'PEAKS', 'Casanovo', 'PepNet'],
        label='PowerNovo2 identified 12,994 unique peptides \n that were not detected by other de novo tools, '
              '\n accounting for 8% of the unique peptides \n across all datasets (12,994 of 161,969)',
        facecolor="#EF5350"
    )
    upset.plot(fig=fig)
    ax = plt.gca()
    ax.grid(False)
    ax.yaxis.label.set_size(16)
    ax.legend_.set_frame_on(False)



    output_file = file_path.replace('csv', 'png')
    plt.savefig(output_file)



input_file = "/Data/benchmark/results/denovo_true_peps(len=30).csv"

if __name__ == '__main__':
    plot_upset(input_file)

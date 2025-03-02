import matplotlib.pyplot as plt


data = {'Casanovo': 35.47,
        'PepNet': 2.08,
        'PowerNovo1': 46.28,
        'PowerNovo2': 8.15}


services = list(data.keys())
execution_times = list(data.values())

plt.figure(figsize=(8, 6))  # Размер графика
plt.bar(services, execution_times, color=['olive', 'royalblue',  'teal', 'r', 'navy'])

plt.title('Execution Time Comparison of Services', fontsize=14)  # Заголовок графика
plt.xlabel('DE NOVO SERVICES', fontsize=14)  # Подпись оси X
plt.ylabel('Execution Time (minutes)', fontsize=14)  # Подпись оси Y

for i, value in enumerate(execution_times):
    plt.text(i, value + 0.5, f'{value:.2f}', ha='center', fontsize=14)

plt.xticks(services,  fontsize=14)

for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)

# Упрощаем отображение сетки



plt.savefig('/Data/benchmark/results/denovo_speed.png')

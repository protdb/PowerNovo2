import os
from pathlib import Path

from powernovo2.config.default_config import setup_run_environment
from powernovo2.inference import PWNInference


BENCHMARK_FOLDER = "/Data/benchmark/"

if __name__ == '__main__':
    benchmark_folder = Path(BENCHMARK_FOLDER)
    dataset_folder = benchmark_folder / 'datasets'
    output_folder = benchmark_folder / 'pw2_output'
    output_folder.mkdir(exist_ok=True)

    files = []
    for dirpath, dirnames, filenames in os.walk(dataset_folder):
        files.extend([os.path.join(dirpath, filename) for filename in filenames if filename.endswith('mgf')])

    cfgs = setup_run_environment()
    cfgs['environment']['annotated_spectra'] = True
    pwn = PWNInference(configs=cfgs)
    pwn.load_models()

    for file in files:
        target_folder = output_folder / Path(file).parent.name
        target_folder.mkdir(exist_ok=True)
        pwn.run(input_file=file)





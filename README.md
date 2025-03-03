# PowerNovo2: Generative Flow-Based Deep Learning for De Novo Peptide Sequencing #
<img title="a title" alt="Alt text" src="/images/logo.png">

## Introduction ##
PowerNovo2 represents the next evolution in de novo peptide sequencing tools, specifically designed to overcome the limitations of autoregressive models commonly used in the field. 
Leveraging a **non-autoregressive generative flow-based architecture**, PowerNovo2 accurately identifies amino acid sequences from tandem mass spectrometry data at speeds 4-5 times faster than traditional models.
With superior scalability and accuracy, PowerNovo2 extends its applications to scenarios such as metaproteomics, monoclonal antibody sequencing, and discovery of novel antigens, areas where current database-dependent approaches may fail due to incomplete sequence libraries.

## Key Features
- **Non-Autoregressive Design**: Avoids cascading prediction errors found in autoregressive models.
- **Generative Flow-Based Architecture**: Models complex conditional dependencies between sequence tokens using latent variables, improving prediction accuracy and robustness.
- **Support for Protein Inference**: Assemble peptide sequences into contigs and map them to protein-level insights using customizable FASTA libraries.
- **Enhanced Throughput**: Processes data up to 5x faster compared to autoregressive models like Transformer-based architectures or LSTM-driven methods.
- **Database-Free**: Enables effective sequencing without requiring pre-existing library data
- **Wide-Ranging Applications**: Suitable for metaproteomics, antibody sequencing, and novel antigen detection

## Installation
PowerNovo2 is implemented with Python (tested on 3.9 or above).
```bash
pip install powernovo2
```
or clone the repository and install the required dependencies.
```bash
git clone https://github.com/protdb/PowerNovo2.git
cd PowerNovo2
pip install -r requirements.txt
```
## Usage
The program can be executed from the command line with various arguments:
PowerNovo2 requires an input MGF file (or folder containing multiple MGF files). 
The following arguments can be used when running the pipeline:
```bash
python3 denovo.py <inputs> [options]
```
Arguments and Options
inputs: Path to an MGF file or folder containing .mgf files. (Required)
### Optional Arguments:
    -w / --working_folder: Specify a working folder for downloads, models, ALPS, etc. Default: powernovo2_work
    -o / --output_folder: Path for results. Default: Current directory.
    -b / --batch_size: Batch size for processing. Default: 16.
    -a / --annotated_spectra: Include this flag if the input MGF files are annotated.
    -alps / --use_assembler: Use ALPS assembler for additional protein inference.
    -c / --num_contigs: Number of generated contigs for post-assembly. Default: 600.
    -k / --contigs_kmer: Size of the k-mer for contigs. Default: 7.
    -p / --protein_inference: Perform protein mapping using provided FASTA files.
    -f / --fasta_path: Specify a custom FASTA file for peptide-protein mappings. Default: UP000005640_9606.fasta.

Run with just the input MGF file:
```bash
python3 denovo.py test.mgf
```
Specifying a Working Directory and Output Folder:
```bash
python3 denovo.py test.mgf -w ./work_dir -o ./results
```
Using ALPS and Protein Inference with FASTA adjusting contig parameters:

```bash
python3 denovo.py test.mgf -alps -p -f custom_proteins.fasta -c 20 -k 10
```

**You can also run the pipeline directly from code** See example in [Colab notebook](/examples/powernovo2_example.ipynb)

    from powernovo2.run import run_inference
    
    if __name__ == '__main__':
        run_inference(
                inputs: str, # input file or folder
                working_folder: str = 'pwn_work', # Path to the project's working folder (see above: -w options).
                output_folder: str = '', # Output folder (see above: -o options)
                batch_size: int = 16,
                use_assembler: bool = True, # (see above: -alps options)
                protein_inference: bool = True, # (see above: -infer options)
                fasta_path: str = '',
                num_contigs: int = 20,
                contigs_kmer: = 7,  
                annotated_spectra: bool = False, # If mfg is annotated, then the prediction results will contain the annotation from mgf
                callback_fn: callable = None  # The function to which the prediction results will be returned
        )
        
        
When launched, the program automatically downloads model weights and ALPS peptide assembler. Download data is located on the Figshare resource 10.6084/m9.figshare.25329586  [Models data](https://figshare.com/s/49d21966f8230445f2a4) 
If necessary, you can download them manually  and put them in the working folder [Models data](https://doi.org/10.6084/m9.figshare.28517777.v1) .

### Output: ###
The pipeline output results are represented by the following files:
* Table with predicted peptides and their confidence score.
* Fasta files with assembled contigs.
* Table of mapping peptides into proteins and protein groups.
* Table containing solved proteins and their peptide sequences.

Examples of output results can be viewed at [Output results](examples/denovo_output/example%28HCD_H.Sapience%29)



### Note: ###
To run the ALPS peptide assembler, Java must be installed on the machine. If Java is not installed, try to install.

```bash
    sudo apt install default-jre
```

## Pipeline ##
Below is a simplified representation of the PowerNovo2 workflow:

<img title="a title" alt="Alt text" src="/images/flow.png">


## License ##
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact ##
or any questions or issues, please contact: [mailto:petro2017@mail.ru]

## References
PowerNovo2 is an advancement of the [PowerNovo1 project](https://www.nature.com/articles/s41598-024-65861-0), 
utilizing non-autoregressive peptide sequence modeling based on generative flow technology.

# iMAG-AD

## 1. Description

iMAG-AD (isoform-informed Multi-view Aggregation on Graphs for Alzheimer's
Disease Gene Prioritization) is a multi-view graph learning method for AD gene
prioritization.

The method uses three transcriptomic views: gene expression, isoform
abundance, and relative isoform usage. Each view is represented by a
13-dimensional feature matrix and a view-specific gene graph. The three graph
embeddings are concatenated and used to predict AD-gene association
probabilities.

## 2. Input data

The `data/features` directory contains the three feature matrices:

- `gene_signed_sig.csv`
- `isoform_signed_sig.csv`
- `ratio_signed_sig.csv`

Each file contains 12 differential-evidence features, one for each
dataset-tissue combination. Gene-level features are calculated as:

```text
sign(logFC) * -log10(FDR)
```

For isoform abundance, each isoform first receives the signed evidence:

```text
e = logFC * -log10(FDR)
```

Positive and negative evidence are pooled separately by averaging their three
largest values. The stronger side determines the magnitude and direction of
the gene-level isoform feature.

Relative isoform usage has no gene-level up or down direction because isoform
proportions within a gene sum to one. Its nonnegative feature is:

```text
mean_top3(abs(delta_ratio) * -log10(FDR))
```

For genes with fewer than three isoforms, all available isoforms are used.
The complete differential-result tables are used without an FDR or effect-size
cutoff.

A view-presence indicator is added during data loading, giving 13 input
features for each view.

The `data/graphs` directory contains three weighted gene-gene edge lists:

- `gene_graph.csv`
- `isoform_graph.csv`
- `ratio_graph.csv`

Each graph is constructed as:

```text
A_view = A_PPI + 0.05 * A_coexpression,view
```

Within each dataset, the 50 strongest absolute correlations of each node are
retained. Edges supported by at least two datasets are aggregated, and the
isoform- and ratio-level networks are mapped to gene pairs by taking the
maximum edge weight between their isoforms.

The graph files contain the columns `gene1`, `gene2`, and `weight`. Self-loops
and symmetric adjacency normalization are added by the program.

`data/labels/ad_gene_labels.csv` contains 146 AD-associated genes and 1,144
reference-negative genes. `DATA_MANIFEST.md` records the dimensions and
SHA-256 checksums of all input files.

## 3. Implementation

iMAG-AD is implemented in Python and depends on PyTorch, NumPy, pandas, and
scikit-learn. The tested environment uses Python 3.8 and PyTorch 1.10.2.

The model uses an independent 32-dimensional graph encoder for each view. The
three embeddings are concatenated and passed to a multilayer perceptron.
Training uses class-weighted cross-entropy and Adam with the following
settings:

- epochs: 100
- learning rate: 0.001
- weight decay: 0.0005
- dropout: 0.3
- gradient clipping: 5

The source files are:

- `src/imag.py`: command-line entry
- `src/data.py`: data and graph loading
- `src/layer.py`: neural network layers
- `src/model.py`: iMAG-AD model
- `src/train.py`: training, cross-validation, and gene ranking
- `src/util.py`: random seeds, standardization, and evaluation metrics

The Conda environment can be created with:

```bash
conda env create -f environment.yml
conda activate imag-ad
```

The same Python dependencies are listed in `requirements.txt`.

## 4. Running

Run stratified five-fold cross-validation over random seeds 0-4:

```bash
python src/imag.py cv
```

Run a short test:

```bash
python src/imag.py cv --epochs 1 --seeds 0 --n-splits 2
```

Run one fold of the five-fold partition:

```bash
python src/imag.py cv --seeds 0 --fold 0
```

Train on all labeled genes and rank the unlabeled genes:

```bash
python src/imag.py rank
```

Generated files are written to the `outputs` directory.

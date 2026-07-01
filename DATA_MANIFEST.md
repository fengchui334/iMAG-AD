# Data manifest

All tabular files are comma-separated UTF-8 text. Graphs contain one
undirected gene pair per row; the loader adds the reverse direction and self
loops before symmetric adjacency normalization.

| File | Rows | Data columns | SHA-256 |
|---|---:|---:|---|
| `data/features/gene_signed_sig.csv` | 31,292 | 12 signed-significance features | `1fd36a243e06e52e3dd9ffae341e20b85c728b3fd454d6a14f5bf16099e5ae13` |
| `data/features/isoform_signed_sig.csv` | 31,292 | 12 signed-significance features | `f5e666272e040415c16660bb6884c67cd3af6e29b382d44452b806d34b53a76e` |
| `data/features/ratio_signed_sig.csv` | 15,598 | 12 signed-significance features | `579a4a54b92a76ab054188ca55ffa8b8faf11bf433f04a3c4b58af88c71f16a7` |
| `data/graphs/gene_graph.csv` | 587,329 | 3 | `5b23ddfd355e2f1dac205d2833d7e87cf92791493989eebd1754902c43b37368` |
| `data/graphs/isoform_graph.csv` | 815,370 | 3 | `4edc175a270980cb70f2be73b51f28bf6fd441f7cb5d33c6112b0a1a2480b5d3` |
| `data/graphs/ratio_graph.csv` | 410,745 | 3 | `aa964a856ee0c2131fe641aa821f5645553ed7ec2bc0c2cf17bd5b66c72e7307` |
| `data/labels/ad_gene_labels.csv` | 1,290 | 2 | `c9f944659e124a7bfa0157e0d56ae0986ffc3469aa5442ed74b5c3c05463120b` |

The model appends one view-presence indicator after feature alignment. All
three model input dimensions are therefore 13.

The labels comprise 146 AD-associated genes and 1,144 reference-negative
genes.

# Data manifest

All tabular files are comma-separated UTF-8 text. Graphs contain one
undirected gene pair per row; the loader adds the reverse direction and self
loops before symmetric adjacency normalization.

| File | Rows | Data columns | SHA-256 |
|---|---:|---:|---|
| `data/features/gene_signed_sig.csv` | 31,292 | 12 signed-significance features | `3b8aef9cba56ebdd8af76fe25cc620c4ee92d745b3b83703dc5525bc07b802e5` |
| `data/features/isoform_signed_sig.csv` | 31,292 | 12 dominant-direction top-3 evidence features | `8963179d0c361899845dc2befb4369c8f3d7beec565127220b4aec4c0b79cd69` |
| `data/features/ratio_signed_sig.csv` | 15,598 | 12 nonnegative top-3 usage-evidence features | `7b92c3a753f84c0cb4f9a00084597da7ae3aaf9e9ee856191889e4ab8b9c78b5` |
| `data/graphs/gene_graph.csv` | 587,329 | 3 | `5b23ddfd355e2f1dac205d2833d7e87cf92791493989eebd1754902c43b37368` |
| `data/graphs/isoform_graph.csv` | 815,370 | 3 | `4edc175a270980cb70f2be73b51f28bf6fd441f7cb5d33c6112b0a1a2480b5d3` |
| `data/graphs/ratio_graph.csv` | 410,745 | 3 | `aa964a856ee0c2131fe641aa821f5645553ed7ec2bc0c2cf17bd5b66c72e7307` |
| `data/labels/ad_gene_labels.csv` | 1,290 | 2 | `c9f944659e124a7bfa0157e0d56ae0986ffc3469aa5442ed74b5c3c05463120b` |

The model appends one view-presence indicator after feature alignment. All
three model input dimensions are therefore 13.

The labels comprise 146 AD-associated genes and 1,144 reference-negative
genes.

# Third-party notices

## Deep Graph Library

`model/conv.py` contains a relation-aware graph-attention implementation
adapted from the DGL `GATConv` implementation:

<https://docs.dgl.ai/en/latest/_modules/dgl/nn/pytorch/conv/gatconv.html>

DGL is distributed under the Apache License 2.0. A copy is included at
`licenses/Apache-2.0.txt`. The SGAN-DTI changes add relation embeddings and
edge-aware attention terms.

## Datasets

The dataset matrices are derived from third-party biomedical databases and
benchmark publications. They are not relicensed under MIT. See
[DATASETS.md](DATASETS.md) before redistributing the `datasets/` directory.


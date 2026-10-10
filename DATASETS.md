# Dataset provenance and use notice

The archive contains only the files required by the released training code:
six binary relation matrices plus fixed positive and 1:1 negative fold files
for each dataset. Similarity matrices, raw identifiers, processed caches,
alternative negative ratios, and unrelated feature files are excluded.

## Included benchmarks

### Hetero-A (`datasets/data_luo`)

Hetero-A originates from the DTINet heterogeneous network:

> Luo Y, Zhao X, Zhou J, et al. A network integration approach for
> drug-target interaction prediction and computational drug repositioning from
> heterogeneous information. *Nature Communications*. 2017;8:573.
> <https://doi.org/10.1038/s41467-017-00680-8>

### Hetero-B (`datasets/data`)

Hetero-B is the expanded heterogeneous benchmark described by MHGNN:

> Li M, Cai X, Xu S, et al. Metapath-aggregated heterogeneous graph neural
> network for drug-target interaction prediction. *Briefings in
> Bioinformatics*. 2023;24(1):bbac578.
> <https://doi.org/10.1093/bib/bbac578>

### Zeng's dataset (`datasets/data_zeng`)

This heterogeneous biomedical benchmark is associated with deepDTnet:

> Zeng X, Zhu S, Lu W, et al. Target identification among known drugs by deep
> learning from heterogeneous networks. *Chemical Science*.
> 2020;11(7):1775-1797. <https://doi.org/10.1039/C9SC04336E>

## Licensing warning

No standalone dataset license was present in the local source directories from
which this reproducibility package was assembled. The matrices incorporate
records from external biomedical databases, whose terms can differ. The MIT
license in this repository applies only to SGAN-DTI code and documentation.

Before making the repository public, the uploader must verify that
redistribution of each matrix is permitted by the originating publications and
databases. If redistribution cannot be confirmed, remove `datasets/` from the
public repository and provide an authorised download/preparation script or
access instructions instead. The code accepts external dataset locations via
`--data-dir`.


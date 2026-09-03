# SmaulNative — Transformer Branch

This is the legacy Transformer implementation of SmaulNative.

The branch contains the older Transformer training/merging/tokenizer/data-generation pipeline:

- `Train.py` — Transformer training.
- `Merge.py` — Transformer checkpoint merging.
- `Tokenizer.py` — tokenizer tooling.
- `data_g.py` — dataset/data processing.
- `download_g.py` — dataset download tooling.
- `syntheticdata_g.py` — synthetic data generation.

This branch is retained for reference and historical experiments and is no longer actively updated. New development is happening on the root RWKV-X implementation in `developement` and related development branches.

## Branches

- `main` — primary root implementation.
- `developement` — active development.
- `forced-architecture-training` — explicit RWKV-X architecture configuration.
- `transformer` — this legacy Transformer branch.
- `transformers-based-RWKV` — legacy Transformers-based RWKV branch.

## License

See the repository license terms in the root project.

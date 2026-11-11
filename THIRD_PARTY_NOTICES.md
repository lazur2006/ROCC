# Third-party materials

The original ROCC code is released under the MIT license in `LICENSE`. This does not replace the terms of third-party code, models, or datasets.

The standalone ROCC model uses `sentence-transformers/all-MiniLM-L12-v2` at revision `a50ef00143b4d5391434df20ae11632588ac25be`. Its model card declares Apache-2.0. The ROCC checkpoint contains the fine-tuned encoder as well as the selector head and CRF. Tokenizer and encoder configuration files are copied from that revision. See [the pinned model card](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/blob/a50ef00143b4d5391434df20ae11632588ac25be/README.md) and `licenses/Apache-2.0.txt`. The ROCC modifications concern the selector training and its additional head and CRF parameters.

The runtime depends on Transformers (Apache-2.0), PyTorch (BSD-style license with additional bundled notices), and pytorch-crf (MIT). They are installed as dependencies rather than vendored. Their distributions contain their respective license texts.

The experiment implementation references [IterCQR](https://github.com/YunahJang/IterCQR), [ANCE](https://huggingface.co/castorini/ance-msmarco-passage), and [Pyserini](https://github.com/castorini/pyserini). No upstream IterCQR source or checkpoint is redistributed here. No license file was identified in the IterCQR repository at revision `ed730be7ebb4541bf13ce14e938ca9e02904af48`. The ANCE model card does not specify a model-weight license. Obtain these assets from their authors and check their terms before use or redistribution.

QReCC's [dataset record](https://zenodo.org/records/5543685) declares CC-BY-SA-3.0. This differs from the Apache-2.0 license of Apple's QReCC code repository. TopiOCQA's [split record](https://zenodo.org/records/6151011) and [corpus record](https://zenodo.org/records/6149599) declare CC0, while the [upstream repository license](https://github.com/mcgill-nlp/topiocqa/blob/main/LICENSE) says CC-BY-NC-SA-4.0. These are distinct source declarations, not a claim that every TopiOCQA asset is available under the same terms. Consult the authors if your intended use depends on resolving this difference.

The historical outputs in the original notebooks and the two required corpus-profile JSON files derive from these datasets. Their inclusion does not grant permission to relicense dataset text. Large corpora and indices are not included in this repository.

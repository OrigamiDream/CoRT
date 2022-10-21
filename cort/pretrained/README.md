# CoRT Pre-trained Backbones

The model requires **KorSci-BERT** and **KorSci-ELECTRA** for experiments. 
Those models are not publicly available for everyone, 
therefore you have to request the rights for the model via following links below. 

- [KorSci-BERT](https://aida.kisti.re.kr/data/107ca6f3-ebcb-4a64-87d5-cea412b76daf)
- [KorSci-ELECTRA](https://aida.kisti.re.kr/data/4079fda0-9580-4f7e-8f10-5815d1878a56)

The models had been trained with Korean-written scientific research corpora.

**KorSci-ELECTRA** is free to download to everyone, 
but **KorSci-BERT** isn't.<br>
Therefore you have to request the rights to the download the model.

The pre-trained models are built on TensorFlow 1.0 environment, 
therefore you have to change the codes making fit to TensorFlow 2.0 environment.


### Directory Hierarchy

The **KorSci-BERT** and **KorSci-ELECTRA** directories must be renamed to `korscibert` and `korscielectra` respectively.
The following directory hierarchy is default setup for default config.
```
- korscibert
  - model.ckpt-262500.data-00000-of-00001
  - model.ckpt-262500.index
  - model.ckpt-262500.meta
  - vocab_kisti.txt
  - tokenization_kisti.py
- korscielectra
  - data
    - models
      - korsci_base
        - checkpoint
        - korsci_base.data-00000-of-00001
        - korsci_base.index
        - korsci_base.meta
    - vocab.txt
  - model
    - tokenization.py
```
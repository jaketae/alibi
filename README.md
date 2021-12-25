# ALiBi

PyTorch implementation of [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409).

![alt-text](https://production-media.paperswithcode.com/methods/Screen_Shot_2021-08-31_at_9.34.28_AM.png)

## Quickstart

Clone this repository.

```sh
git clone https://github.com/jaketae/alibi.git
```

Navigate to the cloned directory. You can use the bare-bone ALiBi decoder via

```python
>>> import torch; from alibi import ALiBiConfig, ALiBiTransformer
>>> config  = ALiBiConfig()
>>> model = ALiBiTransformer(config)
>>> x = torch.randn(8, 100, 128)
>>> model(x).shape
torch.Size([8, 100, 128])
```

By default, the model comes with the following parameters:

```python
ALiBiConfig(
    num_layers=6, 
    d_model=256, 
    num_heads=8, 
    max_len=256, 
    dropout=0.1, 
    causal=True, 
    expansion_factor=1
)
```

To use an encoder instead of a decoder, simply toggle `causal=False`. 

## Abstract

> Since the introduction of the transformer model by Vaswani et al. (2017), a fundamental question remains open: how to achieve extrapolation at inference time to longer sequences than seen during training? We first show that extrapolation can be improved by changing the position representation method, though we find that existing proposals do not allow efficient extrapolation. We introduce a simple and efficient method, Attention with Linear Biases (ALiBi), that allows for extrapolation. ALiBi does not add positional embeddings to the word embeddings; instead, it biases the query-key attention scores with a term that is proportional to their distance. We show that this method allows training a 1.3 billion parameter model on input sequences of length 1024 that extrapolates to input sequences of length 2048, achieving the same perplexity as a sinusoidal position embedding model trained on inputs of length 2048, 11% faster and using 11% less memory. ALiBi's inductive bias towards recency allows it to outperform multiple strong position methods on the WikiText-103 benchmark. Finally, we provide analysis of ALiBi to understand why it leads to better performance.

## Citation

```bibtex
@misc{press2021train,
	title        = {Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation},
	author       = {Ofir Press and Noah A. Smith and Mike Lewis},
	year         = 2021,
	eprint       = {2108.12409},
	archiveprefix = {arXiv},
	primaryclass = {cs.CL}
}
```


# Backpropagation for Continual Learning

Continual learning refers to problems where the input distribution and/or target function are non-stationary. This is in contrast to supervised learning problems where both the input distribution and the target function are unchanging. Standard deep learning methods fail in a continual learning setting after the task distribution changes sufficiently; they gradually lose plasiticity until they learn no better than a shallow network.

[Dohare et al. [2022]](https://openreview.net/forum?id=86sEVRfeGYS) propose the continual backprop (CBP) algorithm to learn continually. They hypothesize that a deep neural network with the vanilla backpropogation algorithm is unable to learn new tasks because the benefits of random initialization are absent after the model has learned a few tasks. We propose three variations of CBP and evaluate them on the _slowly changing regression_ problem, where the input is randomly generated bit data and the input distribution is randomly shifted after a certain number of time steps.

This repository was forked from https://github.com/shibhansh/loss-of-plasticity.

## Installation

```
cd loss-of-plasticity
conda env create -f environment.yml
conda activate lop
pip3 install -e .
```

The instructions on how to generate data and run experiments and generate plots can be found at `lop/slowly_changing_regression/plots/online_performance.py`.

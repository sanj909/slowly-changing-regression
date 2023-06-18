# Loss of Plasticity in Deep Continual Learning

This repository was forked from https://github.com/shibhansh/loss-of-plasticity

# Installation

```sh
virtualenv --python=/usr/bin/python3.8 loss-of-plasticity/
source loss-of-plasticity/bin/activate
cd loss-of-plasticity
pip3 install -r requirements.txt
pip3 install -e .
```

Add these lines in your .zshrc

```sh
source PATH_TO_DIR/loss-of-plasticity/lop/bin/activate
export PYTHONPATH=$PATH:PATH_TO_DIR/lop 
```

Alternatively, type

```
cd loss-of-plasticity
conda env create -f environment.yml
conda activate lop
pip3 install -e .
```

The instructions on how to generate data and run experiments and generate plots can be found at `lop/slowly_changing_regression/plots/online_performance.py`.

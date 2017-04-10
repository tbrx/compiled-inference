# Learn proposal distributions for importance sampling and SMC

This code provides a [PyTorch](http://pytorch.org/) implementation of the method described in [this paper](https://arxiv.org/abs/1602.06701) for training neural networks which can be used as proposal distributions for importance sampling or sequential Monte Carlo:

> Paige, B., & Wood, F. (2016). Inference Networks for Sequential Monte Carlo in Graphical Models. In Proceedings of the 33rd International Conference on Machine Learning. JMLR W&CP 48: 3040-3049.

The largest section of re-usable code is an implementation of a conditional variant of [MADE](https://arxiv.org/abs/1502.03509) as a PyTorch module, found in `learn_smc_proposals.cde`.
This can be used to fit a conditional density estimator.
There is a version for real-valued data and a version for binary data.

For an end-to-end usage example, see the [linear regression notebook](notebooks/Linear-Regression.ipynb).
This notebook defines a generative model using [PyMC](https://github.com/pymc-devs/pymc), a non-conjugate regression model.
It then goes through the process of defining a network which will represent the inverse,
training it on samples from the joint distribution, and then using it for inference.

Two more complex examples are implemented in `learn_smc_proposals.examples`, a heirarchical Poisson model and a factorial hidden Markov model; pre-trained weights are included in this repository.
Figures are generated in the [multilevel Poisson](notebooks/Multilevel-Poisson.ipynb) and [factorial HMM](notebooks/Factorial-HMM.ipynb) notebooks.
The Poisson example demonstrates use of [divide-and-conquer SMC](https://arxiv.org/abs/1406.4993) for inference in a multilevel model; the factorial HMM example demonstrates the binary data implementation and the repeated use of a single learned inverse in a particle filtering context.

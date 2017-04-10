# Learn proposal distributions for importance sampling and SMC

This code provides a [PyTorch](http://pytorch.org/) implementation of the method described in [this paper](https://arxiv.org/abs/1602.06701) for training neural networks which can be used as proposal distributions for importance sampling or sequential Monte Carlo:

Paige, B., & Wood, F. (2016). Inference Networks for Sequential Monte Carlo in Graphical Models. In _Proceedings of the 33rd International Conference on Machine Learning_. JMLR W&CP 48: 3040-3049.

The largest section of re-usable code is an implementation of a conditional variant of [MADE](https://arxiv.org/abs/1502.03509) as a PyTorch module, found in `learn_smc_proposals.cde`.
This can be used to fit a conditional density estimator.
There is a version for real-valued data and a version for binary data.

* The [linear regression notebook](notebooks/Linear-Regression.ipynb) provides an end-to-end usage example. This notebook defines a generative model using [PyMC](https://github.com/pymc-devs/pymc), a non-conjugate regression model. It then goes through the process of defining a network which will represent the inverse, training it on samples from the joint distribution, and then using it for inference.

Two more involved examples are implemented in `learn_smc_proposals.examples`; pre-trained weights are included in this repository. Figures and inference are shown in two notebooks:

* A [multilevel Poisson](notebooks/Multilevel-Poisson.ipynb) model demonstrates use of [divide-and-conquer SMC](https://arxiv.org/abs/1406.4993) for inference in a Bayesian heirarchical model;
* A [factorial HMM](notebooks/Factorial-HMM.ipynb) demonstrates the binary data implementation and the repeated use of a single learned inverse in a particle filtering context.
